"""Serving process lifecycle.

Phase 2 deliberately did not manage servers: printing the launch command was
enough when a human was benchmarking one configuration. The optimizer changed
that -- it benchmarks a dozen candidates in sequence, and no one is going to
start and stop a server by hand between each one. This is that deferral coming
due, not a reversal of it.

Kept as thin as possible. A launch is a command template, a readiness poll and a
teardown; AutoDistiller does not try to understand the runtime it started. The
template matters because the command is environment-specific: on Windows the
server runs inside WSL and needs a wrapper, which is a fact about the machine
rather than about vLLM.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import httpx

logger = logging.getLogger(__name__)

DEFAULT_READY_TIMEOUT_S = 900
"""vLLM compiles kernels and captures CUDA graphs on first start. Minutes, not
seconds, and longer on a cold cache."""

POLL_INTERVAL_S = 2.0
SHUTDOWN_GRACE_S = 20
STOP_TIMEOUT_S = 150
"""vLLM takes the engine core down before releasing the port, which can run to
the better part of a minute. Waiting is cheaper than colliding with it."""

ProgressFn = Callable[[str], None]


class ServerError(RuntimeError):
    """The server failed to start, or died while being measured."""


def wsl_path(local_path: str) -> str:
    """Translate a Windows path for a server running inside WSL.

    A launch that crosses into WSL carries the model path with it, and a
    Windows-style ``artifacts`` path means nothing on the other side. Hugging
    Face repo ids are left alone: they are not paths, and only something that
    exists on this filesystem needs translating.
    """
    candidate = Path(local_path)
    if not candidate.exists():
        return local_path  # a hub id, or already a remote path

    resolved = candidate.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        return resolved.as_posix()
    rest = resolved.as_posix()[len(resolved.drive) :].lstrip("/")
    return f"/mnt/{drive}/{rest}"


@dataclass
class LaunchSpec:
    """How to start a serving runtime on this machine.

    ``template`` is formatted with ``model``, ``port``, ``max_model_len`` and
    ``kv_dtype``. Anything else the runtime needs -- environment variables, a
    WSL wrapper, a specific interpreter -- belongs in the template, because it
    describes the machine rather than the candidate.
    """

    template: str
    url: str = "http://localhost:8000"
    port: int = 8000
    ready_timeout_s: int = DEFAULT_READY_TIMEOUT_S
    env: dict[str, str] = field(default_factory=dict)
    shell: bool = True
    path_translator: Callable[[str], str] | None = None
    """Rewrites the model path for the far side of a boundary. See ``wsl_path``."""

    kv_flag_template: str = "--kv-cache-dtype {kv_dtype}"
    """The flag that selects a non-default KV cache type.

    Backend-specific: vLLM takes one ``--kv-cache-dtype``, llama.cpp takes a
    separate type for keys and values. Only used when the dtype is not the
    default, so the common case stays flagless.
    """

    speculative_flag_template: str = "--speculative-config '{config}'"
    """How this runtime is handed a speculative draft. Backend-specific, like
    ``kv_flag_template``, and unused when nothing is speculating."""

    gpu_memory_utilization: float | None = None
    """Fraction of the device the runtime may claim.

    Left to the runtime's own default when unset. vLLM fills to this number
    whatever the model's size, so a fixed value makes measured peak VRAM a
    readout of the flag rather than of the candidate: a 2.47 GiB model and a
    4.11 GiB one both come back at 7.9 GiB on an 8 GiB card, and a VRAM budget
    tighter than the flag can never be satisfied by any candidate.
    """

    max_num_seqs: int | None = None
    """Sequences the server is sized for.

    The memory estimate is computed for a concurrency; unless the launch says
    the same number, the runtime picks its own -- vLLM defaults to 128 -- and
    allocates for that instead, so the estimate describes a server nobody
    started. It also bounds the draft token slots below.
    """

    nested_shell: bool = False
    """Whether the far side re-parses this command with its own shell.

    A WSL launch is ``wsl -d Ubuntu -e bash -lc "..."``: the command reaches
    bash as the contents of a double-quoted string, so any double quote inside
    it has to arrive escaped. Speculative config is JSON, which is nothing but
    double quotes -- unescaped, bash strips them and vLLM is handed
    ``{method: dflash}``, which is not JSON and fails with an error naming
    neither the cause nor the quoting.
    """

    stop_template: str | None = None
    """How to stop the server, when killing the launched process is not enough.

    Terminating a process only reaches its own tree. A launch that crosses a
    boundary -- ``wsl ...`` from Windows, ``ssh ...`` to another host -- leaves
    the real server running on the far side, still holding the GPU, and every
    later candidate then fails for a reason that has nothing to do with it.
    """

    def command_for(
        self,
        model: str,
        *,
        max_model_len: int | None = None,
        kv_dtype: str = "auto",
        speculative_config: str | None = None,
        speculative_tokens: int | None = None,
    ) -> str:
        # kv_dtype is passed as the backend's flag only when it is not the
        # default, so templates stay readable for the common case.
        kv_flag = "" if kv_dtype == "auto" else self.kv_flag_template.format(kv_dtype=kv_dtype)

        util_flag = (
            ""
            if self.gpu_memory_utilization is None
            else f"--gpu-memory-utilization {self.gpu_memory_utilization:.3f}"
        )
        seqs_flag = "" if self.max_num_seqs is None else f"--max-num-seqs {self.max_num_seqs}"

        spec_flag = ""
        if speculative_config:
            config = (
                speculative_config.replace('"', '\\"') if self.nested_shell else speculative_config
            )
            spec_flag = self.speculative_flag_template.format(config=config)

            # Every sequence reserves one slot per draft token plus one for the
            # verified token, taken out of the batched-token budget. vLLM refuses
            # to start when that leaves nothing to schedule -- observed as
            # "max_num_scheduled_tokens is set to -1536", which is
            # 512 - (15 + 1) * 128 with its own defaults. Ask for the draft slots
            # plus a full prefill so the sum cannot go negative.
            if self.max_num_seqs and speculative_tokens:
                slots = (speculative_tokens + 1) * self.max_num_seqs
                budget = slots + (max_model_len or 2048)
                spec_flag += f" --max-num-batched-tokens {budget}"

        if self.path_translator is not None:
            model = self.path_translator(model)
        return self.template.format(
            model=model,
            port=self.port,
            max_model_len=max_model_len or "",
            kv_dtype=kv_dtype,
            kv_flag=kv_flag,
            spec_flag=spec_flag,
            util_flag=util_flag,
            seqs_flag=seqs_flag,
        )


def wait_until_stopped(url: str, *, timeout_s: int = STOP_TIMEOUT_S) -> bool:
    """Wait for the endpoint to stop serving models. False if it never does.

    "Stopped" is the exact negation of the readiness check: anything other than
    a 200 from ``/v1/models``. Treating only a refused connection as stopped is
    wrong, because the port outlives the engine -- after vLLM exits, something
    still answers 8000 with a 404, and a teardown check that waits for silence
    waits forever.
    """
    deadline = time.monotonic() + timeout_s
    endpoint = f"{url.rstrip('/')}/v1/models"

    while time.monotonic() < deadline:
        try:
            if httpx.get(endpoint, timeout=2.0).status_code != 200:
                return True
        except Exception:
            return True
        time.sleep(POLL_INTERVAL_S)
    return False


def _last_words(log: IO[str] | None, *, lines: int = 12) -> str:
    """The tail of a dead server's stderr, for the error that reports its death.

    Read from a file rather than a pipe, deliberately. A pipe holds about 64 KB
    before the writer blocks, and vLLM prints more than that within seconds of
    starting -- so capturing stderr through PIPE and reading it only after the
    process exits deadlocks the very process being waited on. It then hangs
    until the readiness timeout having produced nothing, which is a worse
    failure than the silence it was meant to fix.
    """
    if log is None:
        return ""
    try:
        log.flush()
        log.seek(0)
        tail = [line.rstrip() for line in log.read().splitlines() if line.strip()]
    except Exception:
        return ""
    if not tail:
        return ""
    return "\n  " + "\n  ".join(tail[-lines:])


def check_port_is_free(url: str) -> None:
    """Refuse to launch into a port something else is already answering on.

    Two failure modes, and the quiet one is worse. If the squatter does not
    answer ``/v1/models`` with a 200, readiness polls until it times out and
    then blames the server we launched -- which never got a chance, because on
    WSL2 a Windows process holding the port stops localhost forwarding
    altogether. If the squatter *does* answer with a 200, the benchmark measures
    it and reports the numbers as the model's.

    Anything that responds at all means occupied. A refused connection is what
    free looks like.
    """
    try:
        response = httpx.get(f"{url.rstrip('/')}/v1/models", timeout=3.0)
    except Exception:
        return  # nothing there, which is what we want

    served = ""
    try:
        models = [m.get("id", "?") for m in response.json().get("data", [])]
        served = f" serving {', '.join(models)}" if models else ""
    except Exception:
        pass

    raise ServerError(
        f"{url} is already in use: something answered with HTTP "
        f"{response.status_code}{served}. Stop it, or point the launch at another "
        f"port. Benchmarking whatever is already there would report its numbers "
        f"as the model's."
    )


def wait_until_ready(
    url: str,
    *,
    timeout_s: int,
    process: subprocess.Popen | None = None,
    log: IO[str] | None = None,
) -> None:
    """Poll until the endpoint serves models, or give up.

    Watches the process too: a server that has already exited will never become
    ready, and waiting the full timeout for that is wasted minutes.
    """
    deadline = time.monotonic() + timeout_s
    endpoint = f"{url.rstrip('/')}/v1/models"

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise ServerError(
                f"server exited with code {process.returncode} before becoming ready"
                + _last_words(log)
            )
        try:
            response = httpx.get(endpoint, timeout=3.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)

    raise ServerError(f"server at {url} was not ready within {timeout_s}s")


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill the launched process and everything it spawned.

    Running through a shell means the server is a grandchild, so terminating
    the process we hold reaches the shell and leaves the server running. That is
    the same shape of mistake as launching across a WSL boundary, and it strands
    a process on the GPU just as effectively.
    """
    if sys.platform == "win32":
        # taskkill /T is the only way to reach a grandchild on Windows.
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=SHUTDOWN_GRACE_S,
            )
        return

    with contextlib.suppress(Exception):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def _terminate(process: subprocess.Popen) -> None:
    """Stop a server, escalating if it does not go quietly.

    Leaving one alive would hold the GPU and make the next candidate fail for
    reasons that have nothing to do with the next candidate.
    """
    if process.poll() is not None:
        return

    # The tree goes first, while the parent is still alive to define it.
    # Terminating the parent up front orphans the server: the shell dies, the
    # process it spawned is reparented, and a later tree kill has nothing left
    # to walk. That is how a server survives its own teardown and holds the GPU
    # for the rest of the run.
    _kill_tree(process)
    try:
        process.wait(timeout=SHUTDOWN_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        logger.warning("server did not stop within %ss; killing", SHUTDOWN_GRACE_S)

    with contextlib.suppress(Exception):
        process.kill()
    with contextlib.suppress(Exception):
        process.wait(timeout=SHUTDOWN_GRACE_S)


@contextlib.contextmanager
def serving(
    spec: LaunchSpec,
    model: str,
    *,
    max_model_len: int | None = None,
    kv_dtype: str = "auto",
    speculative_config: str | None = None,
    speculative_tokens: int | None = None,
    progress: ProgressFn | None = None,
) -> Iterator[str]:
    """Start a server, yield its URL, and always shut it down.

    The teardown runs even when the benchmark raises, because the alternative is
    a stranded process holding VRAM for the rest of the optimization.
    """
    # Before spending a minute starting something that can never be reached.
    check_port_is_free(spec.url)

    command = spec.command_for(
        model,
        max_model_len=max_model_len,
        kv_dtype=kv_dtype,
        speculative_config=speculative_config,
        speculative_tokens=speculative_tokens,
    )
    if progress is not None:
        progress(f"starting {model} ({command[:80]}{'...' if len(command) > 80 else ''})")

    env = {**os.environ, **spec.env}
    # A file, not a pipe. Nothing drains the pipe while readiness is polled, so
    # a runtime that prints more than the buffer holds -- which vLLM does within
    # seconds -- would block on its own stderr and never start. See _last_words.
    log = tempfile.TemporaryFile(mode="w+", errors="replace")  # noqa: SIM115 -- closed in finally
    process = subprocess.Popen(
        command if spec.shell else shlex.split(command),
        shell=spec.shell,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log,
        # Its own process group on POSIX, so the whole tree can be signalled.
        start_new_session=sys.platform != "win32",
    )

    try:
        wait_until_ready(spec.url, timeout_s=spec.ready_timeout_s, process=process, log=log)
        if progress is not None:
            progress(f"server ready at {spec.url}")
        yield spec.url
    finally:
        if progress is not None:
            progress("stopping server")
        _terminate(process)
        with contextlib.suppress(Exception):
            log.close()

        if spec.stop_template:
            with contextlib.suppress(Exception):
                subprocess.run(
                    spec.stop_template if spec.shell else shlex.split(spec.stop_template),
                    shell=spec.shell,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=STOP_TIMEOUT_S,
                )

        # Confirm rather than assume: a server that is still answering is still
        # holding the GPU, and the next candidate would fail on its memory.
        if not wait_until_stopped(spec.url):
            logger.warning(
                "%s is still answering after teardown; a later candidate may fail "
                "on VRAM. Set LaunchSpec.stop_template if the server runs across "
                "a process boundary.",
                spec.url,
            )
        time.sleep(2.0)


__all__ = [
    "DEFAULT_READY_TIMEOUT_S",
    "LaunchSpec",
    "ServerError",
    "check_port_is_free",
    "serving",
    "wait_until_ready",
    "wait_until_stopped",
    "wsl_path",
]
