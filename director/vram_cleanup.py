"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")

_RAM_PRESSURE_PERCENT = 82.0
_RAM_RESERVE_MIN = 6 * 1024**3
_RAM_RESERVE_MAX = 12 * 1024**3


def _virtual_memory() -> tuple[int, int, float] | None:
    """Return (total, available, percent) or None when undetectable."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available), float(vm.percent)
    except Exception:
        pass
    try:
        # Linux: /proc/meminfo (deployment target).
        info: dict[str, int] = {}
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key.strip()] = int(rest.split()[0]) * 1024
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        if total <= 0:
            return None
        return total, available, 100.0 * (total - available) / total
    except Exception:
        return None


def ram_under_pressure() -> tuple[bool, str]:
    """Heuristic RAM pressure check for adaptive deep-unload between segments.

    When system memory cannot be detected, assume pressure (safe fallback →
    deep unload) rather than silently keeping models resident forever.
    """
    vm = _virtual_memory()
    if vm is None:
        return True, "RAM usage unknown"
    total, available, percent = vm
    reserve = max(_RAM_RESERVE_MIN, min(_RAM_RESERVE_MAX, int(total * 0.25)))
    if percent >= _RAM_PRESSURE_PERCENT or available < reserve:
        return True, f"RAM {percent:.0f}% used, {available / 1024**3:.1f} GB free"
    return False, ""


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache."""
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
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")


def cleanup_segment_vram_adaptive(*, enabled: bool = True) -> None:
    """Soft-clean every segment boundary; deep-unload only under RAM pressure.

    Keeps models resident (fast segment turnaround) while still guaranteeing a
    clean pinned-memory pool before pressure causes allocator/hostbuf failures.
    """
    pressured, reason = ram_under_pressure() if enabled else (False, "")
    if enabled and pressured:
        log.info("MiniMax H3 Director: adaptive deep unload triggered (%s)", reason)
    cleanup_segment_vram(enabled=enabled, unload_models=pressured)
