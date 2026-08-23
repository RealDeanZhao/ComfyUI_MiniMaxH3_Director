"""ComfyUI MiniMax H3 Director — timeline plugin for MiniMax-H3 AV generation.

Based on ComfyUI official MiniMax H3 support (PR #15224 / #15228).
Licensed under the Apache License, Version 2.0. See LICENSE.
"""

# docker 里 stdout 默认块缓冲：进程被 OOM killer SIGKILL 时缓冲区日志全部丢失，
# 看起来像"一行日志都没输出就死了"。改为行缓冲，保证每条日志立即落盘。
# （容器侧仍建议加 PYTHONUNBUFFERED=1 双保险。）
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from .nodes.conditioning import (
    MiniMaxH3DirectorConditioning,
    MiniMaxH3DirectorPlannerConditioning,
)
from .nodes.director import MiniMaxH3Director
from .nodes.director_refine import MiniMaxH3DirectorRefine
from .nodes.director_groups import (
    MiniMaxH3DirectorGroupImageToVideo,
    MiniMaxH3DirectorGroupReferenceToVideo,
    MiniMaxH3DirectorGroupsCombine,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Director": MiniMaxH3Director,
    "MiniMaxH3DirectorRefine": MiniMaxH3DirectorRefine,
    # Legacy type id kept so older workflows still load.
    "ComfyMiniMaxH3Director": MiniMaxH3Director,
    "MiniMaxH3DirectorConditioning": MiniMaxH3DirectorConditioning,
    "MiniMaxH3DirectorPlannerConditioning": MiniMaxH3DirectorPlannerConditioning,
    "MiniMaxH3DirectorGroupImageToVideo": MiniMaxH3DirectorGroupImageToVideo,
    "MiniMaxH3DirectorGroupReferenceToVideo": MiniMaxH3DirectorGroupReferenceToVideo,
    # Must stay in NODE_CLASS_MAPPINGS: ComfyUI skips comfy_entrypoint when
    # NODE_CLASS_MAPPINGS is present (if/elif in load_custom_node).
    "MiniMaxH3DirectorGroupsCombine": MiniMaxH3DirectorGroupsCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Director": "MiniMaxH3Director",
    "MiniMaxH3DirectorRefine": "MiniMax H3 Director Refine",
    "ComfyMiniMaxH3Director": "MiniMaxH3Director",
    "MiniMaxH3DirectorConditioning": "MiniMax H3 Director Conditioning",
    "MiniMaxH3DirectorPlannerConditioning": "MiniMax H3 Director Planner Conditioning",
    "MiniMaxH3DirectorGroupImageToVideo": "MiniMax H3 Director Group (Image to Video)",
    "MiniMaxH3DirectorGroupReferenceToVideo": "MiniMax H3 Director Group (Reference to Video)",
    "MiniMaxH3DirectorGroupsCombine": "MiniMax H3 Director Groups Combine",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-MiniMaxH3-Director")

# ── 内存看门狗 ────────────────────────────────────────────────────────────
# 后台线程周期采样 RSS / cgroup 用量 / VRAM。只在内存跳变 ≥ 阈值时输出一行，
# 因此无论进程死在哪一步（包括 ComfyUI 核心的模型加载阶段），日志里都会留下
# 完整的内存增长曲线，用于定位 OOM。
try:
    import threading
    import time as _time

    from .director.vram_cleanup import available_system_memory_mb as _avail_mb

    _WATCHDOG_STOP = threading.Event()
    _watchdog_last = {"rss": None, "vram": None}

    def _watchdog_rss_mb() -> int | None:
        try:
            with open("/proc/self/status", "r") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except Exception:
            pass
        return None

    def _watchdog_vram_mb() -> int:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated()) // (1024 * 1024)
        except Exception:
            pass
        return 0

    def _memory_watchdog() -> None:
        log = logging.getLogger("ComfyUI-MiniMaxH3-Director.watchdog")
        while not _WATCHDOG_STOP.wait(2.0):
            rss = _watchdog_rss_mb()
            vram = _watchdog_vram_mb()
            if rss is None:
                continue
            prev_rss, prev_vram = _watchdog_last["rss"], _watchdog_last["vram"]
            jump = prev_rss is None or abs(rss - prev_rss) >= 512 or abs(vram - (prev_vram or 0)) >= 512
            if jump:
                avail = _avail_mb()
                extra = f", cgroup可用={avail}MB" if avail is not None else ""
                log.info(
                    "[watchdog] RSS=%dMB (%+d), VRAM=%dMB%s",
                    rss, (rss - prev_rss) if prev_rss is not None else 0,
                    vram, extra,
                )
                _watchdog_last["rss"] = rss
                _watchdog_last["vram"] = vram

    if not _WATCHDOG_STOP.is_set() and not hasattr(sys, "_minimax_watchdog_started"):
        sys._minimax_watchdog_started = True
        threading.Thread(
            target=_memory_watchdog, name="minimax-mem-watchdog", daemon=True
        ).start()
        _log.info("MiniMax H3 Director memory watchdog started (RSS/VRAM, 2s poll)")
except Exception as _watchdog_exc:
    _log.warning("MiniMax H3 Director memory watchdog failed to start: %s", _watchdog_exc)

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "MiniMax H3 Director HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /minimax/director/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("MiniMax H3 Director HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
