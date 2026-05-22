"""Wall-clock and RSS memory helpers for benchmark logs ([BENCHMARK] lines)."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Optional

_BENCHMARK_RE = re.compile(
    r"\[BENCHMARK\]\s+(.+)$",
    re.MULTILINE,
)
_KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")


def _try_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError:
        return None


def current_memory_mb(pid: Optional[int] = None) -> Dict[str, float]:
    psutil = _try_psutil()
    if psutil is None:
        return {}
    try:
        proc = psutil.Process(int(pid or os.getpid()))
        mi = proc.memory_info()
        return {
            "rss_mb": float(mi.rss) / (1024.0 * 1024.0),
            "vms_mb": float(mi.vms) / (1024.0 * 1024.0),
        }
    except Exception:
        return {}


class PeakMemoryTracker:
    """Sample RSS during a run; log peak for benchmark parsers."""

    def __init__(self) -> None:
        self._peak_rss_mb = 0.0
        self._last: Dict[str, float] = {}

    def sample(self, pid: Optional[int] = None) -> Dict[str, float]:
        snap = current_memory_mb(pid)
        if snap:
            self._last = dict(snap)
            self._peak_rss_mb = max(self._peak_rss_mb, float(snap.get("rss_mb", 0.0)))
        return dict(self._last)

    @property
    def peak_rss_mb(self) -> float:
        return float(self._peak_rss_mb)

    @property
    def last_rss_mb(self) -> float:
        return float(self._last.get("rss_mb", 0.0))


def log_benchmark(
    *,
    role: str,
    phase: str,
    wall_sec: Optional[float] = None,
    memory: Optional[PeakMemoryTracker] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a single parseable benchmark line to stdout (captured in headless logs)."""
    parts = [f"role={role}", f"phase={phase}"]
    if wall_sec is not None:
        parts.append(f"wall_sec={float(wall_sec):.6f}")
    if memory is not None:
        memory.sample()
        parts.append(f"rss_mb={memory.last_rss_mb:.2f}")
        parts.append(f"peak_rss_mb={memory.peak_rss_mb:.2f}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    print("[BENCHMARK] " + " ".join(parts), flush=True)


def parse_benchmark_lines(log_text: str) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for m in _BENCHMARK_RE.finditer(log_text or ""):
        row: Dict[str, str] = {}
        for km in _KV_RE.finditer(m.group(1)):
            row[km.group(1)] = km.group(2)
        if row:
            rows.append(row)
    return rows


def parse_legacy_timings(log_text: str) -> Dict[str, float]:
    """Fallback: scrape legacy timing lines if [BENCHMARK] is absent."""
    out: Dict[str, float] = {}
    patterns = [
        (r"Training Time:\s*([0-9.]+)s", "training_sec"),
        (r"Dataset Share Prep Time:\s*([0-9.]+)s", "dataset_prep_sec"),
        (r"Client Distribution Time:\s*([0-9.]+)s", "client_distribution_sec"),
        (r"Client Eval Time:\s*([0-9.]+)s", "client_eval_sec"),
        (r"Client Eval Upload Time:\s*([0-9.]+)s", "client_eval_upload_sec"),
        (r"Prover Time:\s*([0-9.]+)s", "prover_sec"),
    ]
    for pat, key in patterns:
        hits = re.findall(pat, log_text or "")
        if hits:
            out[key] = float(hits[-1])
    return out


def summarize_run_logs(
    *,
    node_logs: Dict[int, str],
    client_log: str,
    wall_sec: float,
) -> Dict[str, Any]:
    """Build timing/memory summary for start_all_nodes Run summary."""
    summary: Dict[str, Any] = {"wall_sec": float(wall_sec)}
    peak_rss = 0.0
    timings: Dict[str, float] = {}

    for nid, text in sorted(node_logs.items()):
        for row in parse_benchmark_lines(text):
            phase = row.get("phase", f"node{nid}")
            if "wall_sec" in row:
                timings[f"node{nid}_{phase}_sec"] = float(row["wall_sec"])
            if "peak_rss_mb" in row:
                peak_rss = max(peak_rss, float(row["peak_rss_mb"]))
            if "rss_mb" in row:
                peak_rss = max(peak_rss, float(row["rss_mb"]))
        legacy = parse_legacy_timings(text)
        for k, v in legacy.items():
            timings[f"node{nid}_{k}"] = v
            if k == "training_sec":
                timings.setdefault("training_sec", v)

    for row in parse_benchmark_lines(client_log):
        phase = row.get("phase", "client")
        if "wall_sec" in row:
            timings[f"client_{phase}_sec"] = float(row["wall_sec"])
        if "peak_rss_mb" in row:
            peak_rss = max(peak_rss, float(row["peak_rss_mb"]))
    for k, v in parse_legacy_timings(client_log).items():
        timings[f"client_{k}"] = v

    summary["timings"] = timings
    summary["peak_rss_mb"] = peak_rss if peak_rss > 0 else None
    return summary
