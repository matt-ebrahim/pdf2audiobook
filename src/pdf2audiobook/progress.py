"""Progress reporting — prints stage, chapter, ETA, and timing breakdown."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable


class Progress:
    """Progress reporter with per-stage timing breakdown."""

    def __init__(self, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self._stage_start: float = 0
        self._chapter_times: list[float] = []
        self._stage_timings: dict[str, float] = {}
        self._current_stage: str = ""
        self._pipeline_start: float = time.time()
        self._on_event = on_event
        self._lock = threading.Lock()

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event, data)

    def stage(self, name: str) -> None:
        # Record time for the previous stage
        with self._lock:
            if self._current_stage:
                self._stage_timings[self._current_stage] = time.time() - self._stage_start
            self._current_stage = name
            self._stage_start = time.time()
            self._chapter_times.clear()
        print(f"\n{'='*60}")
        print(f"  STAGE: {name}")
        print(f"{'='*60}")
        self._emit("stage", {"name": name})

    def chapter(self, index: int, total: int, title: str) -> None:
        eta = self._estimate_eta(index, total)
        eta_str = f" | ETA: {eta}" if eta else ""
        print(f"  [{index + 1}/{total}] {title}{eta_str}")
        sys.stdout.flush()
        self._emit("chapter", {"index": index, "total": total, "title": title})

    def chapter_done(self, index: int, total: int) -> None:
        with self._lock:
            elapsed = time.time() - self._stage_start
            self._chapter_times.append(elapsed)
        self._emit("chapter_done", {"index": index, "total": total})

    def detail(self, msg: str) -> None:
        print(f"    {msg}")
        sys.stdout.flush()
        self._emit("detail", {"message": msg})

    def warn(self, msg: str) -> None:
        print(f"    WARNING: {msg}", file=sys.stderr)
        self._emit("warn", {"message": msg})

    def error(self, msg: str) -> None:
        print(f"    ERROR: {msg}", file=sys.stderr)
        self._emit("error", {"message": msg})

    def done(self, msg: str = "Done") -> None:
        print(f"\n  {msg}")
        self._emit("done", {"message": msg})

    def print_timing_summary(self) -> None:
        """Print a breakdown of time spent in each stage."""
        # Finalize current stage timing
        if self._current_stage and self._current_stage not in self._stage_timings:
            self._stage_timings[self._current_stage] = time.time() - self._stage_start

        total = time.time() - self._pipeline_start

        print(f"\n{'='*60}")
        print(f"  TIMING BREAKDOWN")
        print(f"{'='*60}")
        for stage_name, duration in self._stage_timings.items():
            pct = (duration / total * 100) if total > 0 else 0
            print(f"    {stage_name:<30s}  {_fmt_duration(duration):>8s}  ({pct:4.1f}%)")
        print(f"    {'─'*50}")
        print(f"    {'Total':<30s}  {_fmt_duration(total):>8s}")

    def _estimate_eta(self, current: int, total: int) -> str:
        if current == 0 or not self._chapter_times:
            return ""
        avg_time = (time.time() - self._stage_start) / current
        remaining = (total - current) * avg_time
        if remaining < 60:
            return f"{remaining:.0f}s"
        if remaining < 3600:
            return f"{remaining / 60:.1f}min"
        return f"{remaining / 3600:.1f}hr"


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h)}h {int(m)}m"
