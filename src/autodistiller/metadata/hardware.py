"""Hardware detection.

Phase 1 only needs to *record* the hardware a measurement ran on. Phase 2 turns
this into deployment profiling, so the shape of the data is deliberately richer
than Phase 1 strictly requires: compute capability and VRAM are what later
phases filter candidates on.
"""

from __future__ import annotations

import os
import platform

from pydantic import BaseModel, Field

from .hashing import hash_obj

BYTES_PER_GIB = 1024**3


class GPUInfo(BaseModel):
    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str | None = None
    driver_version: str | None = None
    multi_processor_count: int | None = None

    @property
    def total_memory_gib(self) -> float:
        return self.total_memory_bytes / BYTES_PER_GIB


class HardwareInfo(BaseModel):
    """A snapshot of the machine a run happened on."""

    hostname: str
    os: str
    cpu: str
    cpu_count: int | None = None
    total_ram_bytes: int | None = None
    accelerator: str = "cpu"  # "cuda" | "mps" | "cpu"
    gpus: list[GPUInfo] = Field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """Identity of the hardware, ignoring volatile fields like hostname."""
        return hash_obj(
            {
                "accelerator": self.accelerator,
                "gpus": [
                    {
                        "name": g.name,
                        "total_memory_bytes": g.total_memory_bytes,
                        "compute_capability": g.compute_capability,
                    }
                    for g in self.gpus
                ],
                "cpu": self.cpu,
            }
        )

    def describe(self) -> str:
        if self.gpus:
            g = self.gpus[0]
            extra = f" (sm_{g.compute_capability.replace('.', '')})" if g.compute_capability else ""
            return f"{g.name}, {g.total_memory_gib:.1f} GiB VRAM{extra}"
        return f"{self.cpu} (no CUDA device)"


def _driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            raw = pynvml.nvmlSystemGetDriverVersion()
            return raw.decode() if isinstance(raw, bytes) else str(raw)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _total_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _cpu_name() -> str:
    name = platform.processor() or platform.machine()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return name or "unknown"


def detect_hardware() -> HardwareInfo:
    info = HardwareInfo(
        hostname=platform.node(),
        os=platform.platform(),
        cpu=_cpu_name(),
        cpu_count=os.cpu_count(),
        total_ram_bytes=_total_ram(),
    )

    try:
        import torch
    except ImportError:
        return info

    if torch.cuda.is_available():
        info.accelerator = "cuda"
        driver = _driver_version()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info.gpus.append(
                GPUInfo(
                    index=index,
                    name=props.name,
                    total_memory_bytes=int(props.total_memory),
                    compute_capability=f"{props.major}.{props.minor}",
                    driver_version=driver,
                    multi_processor_count=getattr(props, "multi_processor_count", None),
                )
            )
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        info.accelerator = "mps"

    return info


def current_vram_bytes(device_index: int = 0) -> tuple[int, int] | None:
    """Return ``(free, total)`` VRAM for a device, or None when unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(device_index)
        return int(free), int(total)
    except Exception:
        return None


_NVML_READY: bool | None = None


def _nvml_handle(device_index: int):
    """Initialize NVML once and return a device handle, or None.

    The cached flag records whether *initialization* worked, and nothing else. A
    failed handle lookup -- a device index that does not exist, a transient
    driver error -- must not disable NVML for the rest of the process: the
    fallback below reads only the calling process's own CUDA context, so a
    poisoned flag turns every later VRAM reading into a number that cannot see
    the server being benchmarked, which is the whole reason NVML is preferred.
    """
    global _NVML_READY
    try:
        import pynvml
    except Exception:
        _NVML_READY = False
        return None

    if _NVML_READY is None:
        try:
            pynvml.nvmlInit()
            _NVML_READY = True
        except Exception:
            _NVML_READY = False
    if not _NVML_READY:
        return None

    try:
        return pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:
        return None


def device_vram_bytes(device_index: int = 0) -> tuple[int, int] | None:
    """Return device-wide ``(free, total)`` VRAM, as ``nvidia-smi`` reports it.

    Prefers NVML over :func:`current_vram_bytes`. ``torch.cuda.mem_get_info``
    only describes the calling process's own CUDA context, so it cannot see
    memory held by a server in another process -- or, on Windows, by a vLLM
    instance inside the WSL guest. Benchmarking a serving runtime is exactly
    that case, and the torch reading silently under-reports it.
    """
    handle = _nvml_handle(device_index)
    if handle is not None:
        try:
            import pynvml

            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free), int(info.total)
        except Exception:
            pass
    return current_vram_bytes(device_index)
