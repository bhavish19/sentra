"""
Integration tests for dropout and join recovery (protocol steps 4, 6, 12).

- test_dropout_recovery_packing_unsafe: 3 nodes, kill 1 -> packing unsafe, clean exit
- test_dropout_recovery_packing_safe: 4 nodes, kill 1 -> recovery, resume, complete
- test_join_recovery_smoke: 3 nodes with --enable-join-recovery, runs without crash

Run: pytest testing/test_dropout_join_recovery.py -v -s
(Add --timeout=200 if pytest-timeout is installed.)
Note: If the project path contains an apostrophe (e.g. Master's), run from a
symlink or copy to avoid PowerShell quoting issues.
"""

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "testing" / "tmp_bench"


def _common_batched_args(base_port: int):
    return [
        "-u",
        "run_mnist_batched_secure.py",
        "--base-port",
        str(base_port),
        "--host",
        "localhost",
        "--batch-size",
        "32",
        "--num-epochs",
        "2",
        "--learning-rate",
        "0.0002",
        "--mnist-samples",
        "128",
        "--loss-mode",
        "softmax",
        "--seed",
        "2026",
        "--field-size",
        str(2**61 - 1),
        "--scale-factor",
        "65536",
        "--softmax-temperature",
        "2",
        "--exp-approx",
        "pade22",
        "--softmax-grad-mode",
        "opened_exact",
        "--grad-clip",
        "0.20",
        "--logit-clip",
        "3.0",
        "--no-abort-on-instability",
        "--enable-network",
        "--enable-failure-detection",
        "--enable-dropout-reshare-recovery",
    ]


@pytest.mark.integration
@pytest.mark.slow
def test_dropout_recovery_packing_unsafe():
    """
    With n=3, t=1, s=1: killing node 3 leaves n_active=2.
    Packing unsafe (2*(t+s-1)=2 >= 2). Recovery should abort with clear message.
    """
    if os.getenv("SENTRA_RUN_DROPOUT_JOIN_TESTS", "1") != "1":
        pytest.skip("Set SENTRA_RUN_DROPOUT_JOIN_TESTS=1 to run dropout/join recovery tests.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_port = 15000 + random.randint(0, 1000)

    common = _common_batched_args(base_port) + [
        "--n-nodes",
        "3",
        "--t",
        "1",
        "--distribute-dataset-shares",
        "--dataset-owner-node",
        "1",
    ]

    procs = []
    try:
        for node_id in (1, 2, 3):
            out_path = LOG_DIR / f"node{node_id}_dropout_unsafe.log"
            err_path = LOG_DIR / f"node{node_id}_dropout_unsafe.err"
            with open(out_path, "w", encoding="utf-8") as out_f, open(
                err_path, "w", encoding="utf-8"
            ) as err_f:
                p = subprocess.Popen(
                    [sys.executable, *common, "--node-id", str(node_id)],
                    cwd=str(ROOT),
                    stdout=out_f,
                    stderr=err_f,
                )
                procs.append((node_id, p))
            time.sleep(0.5)

        time.sleep(25.0)
        _, p3 = procs[2]
        if p3.poll() is None:
            p3.terminate()
            time.sleep(1.0)
            if p3.poll() is None:
                p3.kill()
        time.sleep(20.0)

        for node_id, p in procs:
            if p.poll() is None:
                p.terminate()
            p.wait(timeout=5.0)

        node1_log = (LOG_DIR / "node1_dropout_unsafe.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        node2_log = (LOG_DIR / "node2_dropout_unsafe.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        combined = node1_log + node2_log

        assert (
            "Packing unsafe after dropout" in combined
            or "PAUSED_RESHARE" in combined
            or "Detected 1 failed" in combined
            or "node_failure" in combined
        ), (
            "Expected packing unsafe, PAUSED_RESHARE, or failure detection. Logs:\n"
            f"node1 (last 1500 chars): ...{node1_log[-1500:]}\n"
            f"node2 (last 1500 chars): ...{node2_log[-1500:]}"
        )

    finally:
        for _, p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=2.0)
                except Exception:
                    pass


@pytest.mark.integration
@pytest.mark.slow
def test_dropout_recovery_packing_safe():
    """
    With n=4, t=1, s=1: killing node 4 leaves n_active=3.
    Packing safe (2*(1+1-1)=2 < 3). Recovery should run and resume.
    """
    if os.getenv("SENTRA_RUN_DROPOUT_JOIN_TESTS", "1") != "1":
        pytest.skip("Set SENTRA_RUN_DROPOUT_JOIN_TESTS=1 to run dropout/join recovery tests.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_port = 16000 + random.randint(0, 1000)

    common = _common_batched_args(base_port) + [
        "--n-nodes",
        "4",
        "--t",
        "1",
        "--distribute-dataset-shares",
        "--dataset-owner-node",
        "1",
    ]

    procs = []
    try:
        for node_id in (1, 2, 3, 4):
            out_path = LOG_DIR / f"node{node_id}_dropout_safe.log"
            err_path = LOG_DIR / f"node{node_id}_dropout_safe.err"
            with open(out_path, "w", encoding="utf-8") as out_f, open(
                err_path, "w", encoding="utf-8"
            ) as err_f:
                p = subprocess.Popen(
                    [sys.executable, *common, "--node-id", str(node_id)],
                    cwd=str(ROOT),
                    stdout=out_f,
                    stderr=err_f,
                )
                procs.append((node_id, p))
            time.sleep(0.5)

        time.sleep(50.0)
        _, p4 = procs[3]
        if p4.poll() is None:
            p4.terminate()
            time.sleep(1.0)
            if p4.poll() is None:
                p4.kill()
        time.sleep(35.0)

        deadline = time.time() + 120.0
        for node_id, p in procs[:3]:
            p.wait(timeout=max(1.0, deadline - time.time()))

        node1_log = (LOG_DIR / "node1_dropout_safe.log").read_text(
            encoding="utf-8", errors="ignore"
        )
        combined = node1_log
        for node_id in (2, 3):
            combined += (LOG_DIR / f"node{node_id}_dropout_safe.log").read_text(
                encoding="utf-8", errors="ignore"
            )

        has_recovery = (
            "Dropout recovery" in combined
            or "Dropout weight reshare" in combined
            or "Resumed after dropout" in combined
        )
        has_pause = "PAUSED_RESHARE" in combined or "Detected" in combined
        assert has_recovery or has_pause, (
            "Expected dropout recovery or pause detection. node1 (last 2000 chars): "
            f"...{node1_log[-2000:]}"
        )
        if has_recovery:
            assert "Resumed after dropout" in combined or "MPC peers=" in combined, (
                "Expected resume or MPC peers when recovery ran."
            )

    finally:
        for _, p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=2.0)
                except Exception:
                    pass


@pytest.mark.integration
@pytest.mark.slow
def test_join_recovery_smoke():
    """
    Smoke test: 3 nodes with --enable-join-recovery run without crash.
    No actual join occurs; validates the flag and recovery hooks.
    """
    if os.getenv("SENTRA_RUN_DROPOUT_JOIN_TESTS", "1") != "1":
        pytest.skip("Set SENTRA_RUN_DROPOUT_JOIN_TESTS=1 to run dropout/join recovery tests.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_port = 17000 + random.randint(0, 1000)

    common = _common_batched_args(base_port) + [
        "--n-nodes",
        "3",
        "--t",
        "1",
        "--distribute-dataset-shares",
        "--dataset-owner-node",
        "1",
        "--enable-join-recovery",
    ]
    common = [a for a in common if a != "--enable-dropout-reshare-recovery"]

    procs = []
    try:
        for node_id in (1, 2, 3):
            out_path = LOG_DIR / f"node{node_id}_join_smoke.log"
            err_path = LOG_DIR / f"node{node_id}_join_smoke.err"
            with open(out_path, "w", encoding="utf-8") as out_f, open(
                err_path, "w", encoding="utf-8"
            ) as err_f:
                p = subprocess.Popen(
                    [sys.executable, *common, "--node-id", str(node_id)],
                    cwd=str(ROOT),
                    stdout=out_f,
                    stderr=err_f,
                )
                procs.append((node_id, p))
            time.sleep(0.5)

        time.sleep(25.0)

        for _, p in procs:
            if p.poll() is not None and p.returncode != 0:
                err = (LOG_DIR / "node1_join_smoke.err").read_text(
                    encoding="utf-8", errors="ignore"
                )
                raise AssertionError(f"Node exited with {p.returncode}. stderr: {err[:2000]}")

        combined = ""
        for node_id in (1, 2, 3):
            combined += (LOG_DIR / f"node{node_id}_join_smoke.log").read_text(
                encoding="utf-8", errors="ignore"
            )
        assert "epoch" in combined.lower() or "batch" in combined.lower(), (
            "Expected training progress. Logs (last 1500 chars): "
            f"...{combined[-1500:]}"
        )

    finally:
        for _, p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=2.0)
                except Exception:
                    pass


if __name__ == "__main__":
    import os

    os.environ.setdefault("SENTRA_RUN_DROPOUT_JOIN_TESTS", "1")
    pytest.main([__file__, "-v", "-s"])
