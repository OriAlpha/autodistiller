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
