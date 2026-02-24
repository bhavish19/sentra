import random
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_batched_stage_invariants():
    """
    Runs a tiny 3-node secure-batched MNIST job and checks stage probes to
    localize numerical instability:
    - logits probe should stay bounded
    - dz2 probe should be O(1) for CE softmax gradients
    - probs_est probe should be finite, near simplex
    """
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "testing" / "tmp_bench"
    log_dir.mkdir(parents=True, exist_ok=True)

    base_port = 14000 + random.randint(0, 1000)
    common_args = [
        "-u",
        "run_mnist_batched_secure.py",
        "--n-nodes",
        "3",
        "--t",
        "1",
        "--enable-network",
        "--base-port",
        str(base_port),
        "--host",
        "localhost",
        "--batch-size",
        "64",
        "--num-epochs",
        "1",
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
        "--debug-numerics",
        "--grad-norm-threshold",
        "1500",
        "--explode-logit-threshold",
        "1000000000",
    ]

    procs = []
    try:
        for node_id in (1, 2, 3):
            out_path = log_dir / f"node{node_id}_stageinv.log"
            err_path = log_dir / f"node{node_id}_stageinv.err"
            with open(out_path, "w", encoding="utf-8") as out_f, open(err_path, "w", encoding="utf-8") as err_f:
                p = subprocess.Popen(
                    [sys.executable, *common_args, "--node-id", str(node_id)],
                    cwd=str(root),
                    stdout=out_f,
                    stderr=err_f,
                )
                procs.append(p)
            time.sleep(0.4)

        deadline = time.time() + 420.0
        for p in procs:
            p.wait(timeout=max(1.0, deadline - time.time()))
        for p in procs:
            assert p.returncode == 0, f"Node process failed with code {p.returncode}"

        node1_log = (log_dir / "node1_stageinv.log").read_text(encoding="utf-8", errors="ignore")
        node1_err = (log_dir / "node1_stageinv.err").read_text(encoding="utf-8", errors="ignore")
        assert "Traceback" not in node1_log
        assert "Traceback" not in node1_err

        m_logits = re.search(r"Epoch 1 Probe logits\[0\]\[:10\]: \[(.*?)\]", node1_log, re.DOTALL)
        m_dz2 = re.search(r"Epoch 1 Probe dz2\[0\]\[:10\]: \[(.*?)\]", node1_log, re.DOTALL)
        m_probs = re.search(
            r"Epoch 1 Probe probs_est stats: sum=([0-9eE+\-.]+), min=([0-9eE+\-.]+), max=([0-9eE+\-.]+), target_sum=([0-9eE+\-.]+)",
            node1_log,
        )
        assert m_logits and m_dz2 and m_probs, "Missing stage probes in node 1 log"

        logits_vals = [float(v.strip()) for v in m_logits.group(1).split(",")]
        dz2_vals = [float(v.strip()) for v in m_dz2.group(1).split(",")]
        probs_sum = float(m_probs.group(1))
        probs_min = float(m_probs.group(2))
        probs_max = float(m_probs.group(3))

        max_abs_logit_probe = max(abs(v) for v in logits_vals)
        max_abs_dz2_probe = max(abs(v) for v in dz2_vals)

        # Invariant expectations for stable CE/softmax region.
        assert max_abs_logit_probe < 100.0, f"logit probe exploded: {max_abs_logit_probe:.6f}"
        assert max_abs_dz2_probe < 5.0, f"dz2 probe exploded: {max_abs_dz2_probe:.6f}"
        assert 0.95 <= probs_sum <= 1.05, f"probs_est sum drifted: {probs_sum:.6f}"
        assert probs_min >= -0.05, f"probs_est min invalid: {probs_min:.6f}"
        assert probs_max <= 1.05, f"probs_est max invalid: {probs_max:.6f}"

    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

