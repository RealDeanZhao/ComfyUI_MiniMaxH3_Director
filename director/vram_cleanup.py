"""Release GPU + system memory between MiniMax H3 Director segment runs.

System-RAM notes (docker --memory OOM):
- torch/glibc do not eagerly return freed multi-GB blocks to the OS; RSS climbs
  monotonically across segments until the cgroup limit kills the process.
  ``malloc_trim(0)`` walks all arenas and releases free heap back to the kernel.
- ``available_system_memory_mb`` is cgroup-aware: inside a container,
  /proc/meminfo shows the *host* value, so the cgroup limit/current are read
  and the tighter of the two is used.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")

# 段前预检阈值：低于此值先做一次系统内存回收，仍不足则报错而非被内核 SIGKILL。
MIN_FREE_SYSTEM_MEMORY_MB = 8 * 1024


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _read_meminfo_mb(key: str) -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _cgroup_memory_mb() -> tuple[int | None, int | None]:
    """Return (limit_mb, current_mb) from cgroup v2 then v1; None when unlimited."""
    limit = _read_int("/sys/fs/cgroup/memory.max")
    current = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None or current is None:
        limit_v1 = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        current_v1 = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if limit_v1 is not None and current_v1 is not None and limit_v1 < (1 << 60):
            return limit_v1 // (1024 * 1024), current_v1 // (1024 * 1024)
        return None, None
    if limit >= (1 << 60):  # "max"
        return None, current // (1024 * 1024)
    return limit // (1024 * 1024), current // (1024 * 1024)


def available_system_memory_mb() -> int | None:
    """Best-effort free system memory inside the current cgroup, in MB."""
    host_avail = _read_meminfo_mb("MemAvailable")
    limit, current = _cgroup_memory_mb()
    if limit is not None and current is not None:
        cgroup_avail = max(0, limit - current)
        if host_avail is None:
            return cgroup_avail
        return min(host_avail, cgroup_avail)
    return host_avail


def malloc_trim_heap() -> None:
    """Ask glibc to return free heap (all arenas) to the OS. No-op elsewhere."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def release_system_memory() -> None:
    """gc + glibc heap trim — returns freed CPU tensors' pages to the OS."""
    gc.collect()
    malloc_trim_heap()


def log_memory_snapshot(label: str) -> None:
    """One-line RSS/VRAM footprint for diagnosing per-segment memory growth."""
    try:
        rss = _read_meminfo_mb("VmRSS") or 0
        peak = _read_meminfo_mb("VmHWM") or 0
        vram_alloc = vram_reserved = 0
        try:
            import torch

            if torch.cuda.is_available():
                vram_alloc = int(torch.cuda.memory_allocated()) // (1024 * 1024)
                vram_reserved = int(torch.cuda.memory_reserved()) // (1024 * 1024)
        except Exception:
            pass
        log.info(
            "[memory] %s: RSS=%dMB (peak %dMB), VRAM alloc=%dMB reserved=%dMB",
            label, rss, peak, vram_alloc, vram_reserved,
        )
    except Exception as exc:
        log.debug("memory snapshot failed: %s", exc)


def ensure_system_memory(label: str, min_free_mb: int = MIN_FREE_SYSTEM_MEMORY_MB) -> None:
    """Preflight before a heavy segment: reclaim once, then fail loudly if short.

    Raising a clear ValueError beats the kernel OOM-killer SIGKILL, which loses
    all buffered logs and gives the user nothing to debug with.
    """
    free = available_system_memory_mb()
    if free is None or free >= min_free_mb:
        return
    log.warning(
        "[memory] %s: only %dMB system memory free (<%dMB) — running gc + heap trim.",
        label, free, min_free_mb,
    )
    release_system_memory()
    free = available_system_memory_mb()
    if free is not None and free < min_free_mb:
        raise ValueError(
            f"系统内存不足：{label} 前仅剩 {free}MB 可用（阈值 {min_free_mb}MB）。"
            "多段任务每段只应消耗自己素材；若持续出现，请减小单段时长/分辨率，"
            "或提高容器 --memory 限制。"
            f" / Not enough system memory before {label}: {free}MB free "
            f"(threshold {min_free_mb}MB). Reduce per-segment duration/resolution "
            "or raise the container memory limit."
        )


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache.

    Also trims the glibc heap so decoded-frame buffers freed on CPU are handed
    back to the OS instead of lingering as RSS (critical for multi-segment runs
    under a docker --memory cap).
    """
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    malloc_trim_heap()
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
