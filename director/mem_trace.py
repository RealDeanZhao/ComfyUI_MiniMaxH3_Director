"""Lightweight RSS memory tracing for MiniMax H3 Director runs.

Logs process RSS / peak RSS at key executor points so multi-segment RAM growth
can be attributed to an exact segment + phase from the ComfyUI log alone.
Linux reads /proc/self/status; other platforms fall back to resource/psutil.
"""

from __future__ import annotations

import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.mem")


def _rss_from_proc() -> tuple[float, float] | None:
    try:
        with open("/proc/self/status", "r") as fh:
            cur = peak = None
            for line in fh:
                if line.startswith("VmRSS:"):
                    cur = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    peak = float(line.split()[1]) / 1024.0
                if cur is not None and peak is not None:
                    return cur, peak
    except OSError:
        pass
    return None


def _rss_fallback() -> tuple[float, float]:
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes, Linux reports KiB.
    try:
        import platform

        if platform.system() == "Darwin":
            peak = peak / (1024.0 * 1024.0)
            cur = peak
        else:
            peak = peak / 1024.0
            cur = peak
    except Exception:
        pass
    try:
        import psutil

        cur = float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass
    return cur, peak


def rss_mb() -> tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb)."""
    got = _rss_from_proc()
    if got is not None:
        return got
    return _rss_fallback()


def trace(label: str, *, level: int = logging.INFO) -> str:
    """Log ``[mem] label: rss=…MB peak=…MB`` and return the formatted line."""
    cur, peak = rss_mb()
    line = f"[mem] {label}: rss={cur:.0f}MB peak={peak:.0f}MB"
    log.log(level, "Director %s", line)
    return line


def tensor_bytes_mb(t) -> float:
    """Best-effort size of a tensor / dict-of-tensors in MB."""
    try:
        import torch

        if torch.is_tensor(t):
            return float(t.numel()) * float(t.element_size()) / (1024.0 * 1024.0)
        if isinstance(t, dict):
            total = 0.0
            for v in t.values():
                total += tensor_bytes_mb(v)
            return total
        if isinstance(t, (list, tuple)):
            return sum(tensor_bytes_mb(v) for v in t)
    except Exception:
        pass
    return 0.0
