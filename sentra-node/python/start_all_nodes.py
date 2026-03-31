"""
Python script to start all SENTRA nodes
Usage: python start_all_nodes.py [--use-dp-sgd]
"""

import subprocess
import sys
import time
<<<<<<< HEAD
=======
import argparse
>>>>>>> main
import os
import json
import datetime
import re
from pathlib import Path
import math
<<<<<<< HEAD
from start_all_nodes_cli import parse_and_validate_args
=======
>>>>>>> main

try:
    import psutil
except ImportError:
    psutil = None  # Optional: enables memory stats in headless runs


def _build_node_command(args, node_id):
    """Build command for a node process."""
<<<<<<< HEAD
    cmd = [
        sys.executable,
        '-u',
        'run_mnist_batched_secure.py',
        '--node-id', str(node_id),
        '--n-nodes', str(args.n_nodes),
        '--base-port', str(args.base_port),
        '--host', str(args.host),
        '--batch-size', str(args.batch_size),
        '--num-epochs', str(args.num_epochs),
        '--learning-rate', str(args.learning_rate),
        '--t', str(args.t),
        '--seed', str(args.seed),
        '--mnist-samples', str(args.mnist_samples),
        '--enable-network',
    ]
    cmd.extend([
        '--accum-steps', str(args.accum_steps),
        '--loss-mode', str(args.loss_mode),
        '--scale-factor', str(args.scale_factor),
        '--field-size', str(args.field_size),
        '--softmax-temperature', str(args.softmax_temperature),
        '--grad-clip', str(args.grad_clip),
        '--logit-clip', str(args.logit_clip),
        '--exp-approx', str(args.exp_approx),
        '--softmax-grad-mode', str(args.softmax_grad_mode),
        '--explode-logit-threshold', str(args.explode_logit_threshold),
        '--loss-growth-threshold', str(args.loss_growth_threshold),
        '--grad-norm-threshold', str(args.grad_norm_threshold),
    ])
    if args.distribute_dataset_shares:
        cmd.extend([
            '--distribute-dataset-shares',
            '--dataset-owner-node', str(args.dataset_owner_node),
            '--dataset-distribution-timeout', str(args.dataset_distribution_timeout),
        ])
    if args.receive_dataset_shares_from_client:
        cmd.extend([
            '--receive-dataset-shares-from-client',
            '--dataset-source-node-id', str(args.dataset_source_node_id),
            '--dataset-distribution-timeout', str(args.dataset_distribution_timeout),
        ])
        if args.client_eval_after_training:
            cmd.extend([
                '--client-eval-after-training',
                '--client-eval-samples', str(args.client_eval_samples),
            ])
    if args.export_reconstructed_model:
        cmd.extend([
            '--export-reconstructed-model', str(args.export_reconstructed_model),
            '--export-timeout', str(args.export_timeout),
        ])
    if args.no_abort_on_instability:
        cmd.append('--no-abort-on-instability')
    if args.debug_numerics:
        cmd.append('--debug-numerics')
    if args.debug_division:
        cmd.append('--debug-division')
    if args.packed_forward_pilot:
        cmd.append('--packed-forward-pilot')
    if args.packed_forward_native:
        cmd.append('--packed-forward-native')
    if args.packed_end2end:
        cmd.append('--packed-end2end')
    if args.dpss_refresh_interval > 0:
        cmd.extend(['--dpss-refresh-interval', str(args.dpss_refresh_interval)])
    cmd.extend(['--membership-epoch', str(args.membership_epoch)])
    if args.enable_failure_detection:
        cmd.append('--enable-failure-detection')
    if getattr(args, "enable_dropout_reshare_recovery", False):
        cmd.append('--enable-dropout-reshare-recovery')
    if getattr(args, "enable_join_recovery", False):
        cmd.append('--enable-join-recovery')
    if getattr(args, "use_kvs_dataset", False):
        cmd.append('--use-kvs-dataset')
    if getattr(args, "use_weight_versioning", False):
        cmd.append('--use-weight-versioning')
=======
    if args.dataset == 'mnist':
        script_name = 'run_mnist_batched_secure.py' if args.batched else 'run_mnist_secure.py'
        cmd = [
            sys.executable,
            '-u',
            script_name,
            '--node-id', str(node_id),
            '--n-nodes', str(args.n_nodes),
            '--base-port', str(args.base_port),
            '--host', str(args.host),
            '--batch-size', str(args.batch_size),
            '--num-epochs', str(args.num_epochs),
            '--learning-rate', str(args.learning_rate),
            '--t', str(args.t),
            '--seed', str(args.seed),
            '--mnist-samples', str(args.mnist_samples),
            '--enable-network',
        ]
        if args.batched:
            cmd.extend([
                '--accum-steps', str(args.accum_steps),
                '--loss-mode', str(args.loss_mode),
                '--scale-factor', str(args.scale_factor),
                '--field-size', str(args.field_size),
                '--softmax-temperature', str(args.softmax_temperature),
                '--grad-clip', str(args.grad_clip),
                '--logit-clip', str(args.logit_clip),
                '--exp-approx', str(args.exp_approx),
                '--softmax-grad-mode', str(args.softmax_grad_mode),
                '--explode-logit-threshold', str(args.explode_logit_threshold),
                '--loss-growth-threshold', str(args.loss_growth_threshold),
                '--grad-norm-threshold', str(args.grad_norm_threshold),
            ])
            if args.distribute_dataset_shares:
                cmd.extend([
                    '--distribute-dataset-shares',
                    '--dataset-owner-node', str(args.dataset_owner_node),
                    '--dataset-distribution-timeout', str(args.dataset_distribution_timeout),
                ])
            if args.receive_dataset_shares_from_client:
                cmd.extend([
                    '--receive-dataset-shares-from-client',
                    '--dataset-source-node-id', str(args.dataset_source_node_id),
                    '--dataset-distribution-timeout', str(args.dataset_distribution_timeout),
                ])
                if args.client_eval_after_training:
                    cmd.extend([
                        '--client-eval-after-training',
                        '--client-eval-samples', str(args.client_eval_samples),
                    ])
            if args.export_reconstructed_model:
                cmd.extend([
                    '--export-reconstructed-model', str(args.export_reconstructed_model),
                    '--export-timeout', str(args.export_timeout),
                ])
            if args.no_abort_on_instability:
                cmd.append('--no-abort-on-instability')
            if args.debug_numerics:
                cmd.append('--debug-numerics')
            if args.debug_division:
                cmd.append('--debug-division')
            if args.packed_forward_pilot:
                cmd.append('--packed-forward-pilot')
            if args.packed_forward_native:
                cmd.append('--packed-forward-native')
            if args.packed_end2end:
                cmd.append('--packed-end2end')
            if args.dpss_refresh_interval > 0:
                cmd.extend(['--dpss-refresh-interval', str(args.dpss_refresh_interval)])
            cmd.extend(['--membership-epoch', str(args.membership_epoch)])
            if args.enable_failure_detection:
                cmd.append('--enable-failure-detection')
            if getattr(args, "enable_dropout_reshare_recovery", False):
                cmd.append('--enable-dropout-reshare-recovery')
            if getattr(args, "enable_join_recovery", False):
                cmd.append('--enable-join-recovery')
            if getattr(args, "use_kvs_dataset", False):
                cmd.append('--use-kvs-dataset')
            if getattr(args, "use_weight_versioning", False):
                cmd.append('--use-weight-versioning')
    else:
        cmd = [
            sys.executable,
            '-u',
            'run_node.py',
            '--node-id', str(node_id),
            '--n-nodes', str(args.n_nodes),
            '--base-port', str(args.base_port),
            '--host', str(args.host),
            '--batch-size', str(args.batch_size),
            '--num-epochs', str(args.num_epochs),
            '--learning-rate', str(args.learning_rate),
            '--t', str(args.t),
            '--s', str(args.s),
            '--dataset', str(args.dataset),
            '--seed', str(args.seed),
            '--train-mode', str(args.train_mode),
            '--no-wait',
        ]
>>>>>>> main
    return cmd


def _extract_metrics_from_log(log_text):
    """Parse key metrics from a node log."""
    metrics = {}
    acc_matches = re.findall(r"Epoch\s+(\d+)\s+Test Accuracy:\s+([0-9.]+)%", log_text)
    loss_matches = re.findall(r"Epoch\s+(\d+)\s+Test Loss:\s+([0-9.]+)", log_text)
    recon_matches = re.findall(r"Reconstructed-model MNIST test accuracy:\s+([0-9.]+)", log_text)
    batch_time_matches = re.findall(r"Batch\s+\d+\s+completed in\s+([0-9.]+)s", log_text)
    prover_time_matches = re.findall(
        r"(?:Prover(?:\s+Time)?|proof(?:\s+time)?)\s*[:=]\s*([0-9.]+)\s*s?",
        log_text,
        flags=re.IGNORECASE,
    )
    dataset_prep_matches = re.findall(r"Dataset Share Prep Time:\s*([0-9.]+)s", log_text)
    training_time_matches = re.findall(r"Training Time:\s*([0-9.]+)s", log_text)
    eval_upload_matches = re.findall(r"Client Eval Upload Time:\s*([0-9.]+)s", log_text)
    status_ok = "Training completed successfully!" in log_text
    status_err = "Error occurred" in log_text or "Traceback (most recent call last)" in log_text

    if acc_matches:
        ep, val = acc_matches[-1]
        metrics["last_epoch_acc_epoch"] = int(ep)
        metrics["last_epoch_acc_pct"] = float(val)
    if loss_matches:
        ep, val = loss_matches[-1]
        metrics["last_epoch_loss_epoch"] = int(ep)
        metrics["last_epoch_loss"] = float(val)
    if recon_matches:
        metrics["reconstructed_acc"] = float(recon_matches[-1])
    if batch_time_matches:
        vals = [float(v) for v in batch_time_matches]
        metrics["avg_batch_time_sec"] = round(sum(vals) / len(vals), 4)
        metrics["max_batch_time_sec"] = round(max(vals), 4)
    if prover_time_matches:
        metrics["prover_time_sec_logged"] = float(prover_time_matches[-1])
    if dataset_prep_matches:
        metrics["dataset_share_prep_time_sec"] = float(dataset_prep_matches[-1])
    if training_time_matches:
        metrics["training_time_sec"] = float(training_time_matches[-1])
    if eval_upload_matches:
        metrics["client_eval_upload_time_sec"] = float(eval_upload_matches[-1])
    metrics["status_ok"] = status_ok
    metrics["status_err"] = status_err
    return metrics


def _append_run_to_xlsx(xlsx_path, row_dict):
    """Append one run summary row to an Excel sheet."""
    try:
        from openpyxl import Workbook, load_workbook
    except Exception as exc:
        raise RuntimeError(
            "Excel export requires openpyxl. Install with: pip install openpyxl"
        ) from exc

    headers = [
        "timestamp",
        "run_start",
        "run_end",
        "run_id",
        "status",
        "duration_sec",
        "duration_min",
        "n_nodes",
        "dataset",
        "batched",
        "train_mode",
        "loss_mode",
        "softmax_grad_mode",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "field_size",
        "scale_factor",
        "softmax_temperature",
        "grad_clip",
        "logit_clip",
        "seed",
        "final_epoch_acc_pct",
        "final_epoch_loss",
        "reconstructed_acc",
        "node_success_count",
        "node_elapsed_sec",
        "prover_time_sec",
        "avg_batch_time_sec",
        "max_batch_time_sec",
        "dataset_share_prep_time_sec",
        "training_time_sec",
        "client_distribution_time_sec",
        "client_eval_time_sec",
        "client_eval_upload_time_sec",
        "memory_baseline_mb",
        "memory_peak_mb",
        "memory_overhead_mb",
        "node_memory_baseline_mb",
        "node_memory_peak_mb",
        "node_memory_overhead_mb",
        "node_return_codes",
        "log_dir",
        "command",
    ]
    old_headers = [
        "timestamp",
        "run_id",
        "status",
        "duration_sec",
        "n_nodes",
        "dataset",
        "batched",
        "train_mode",
        "loss_mode",
        "softmax_grad_mode",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "field_size",
        "scale_factor",
        "softmax_temperature",
        "grad_clip",
        "logit_clip",
        "seed",
        "final_epoch_acc_pct",
        "final_epoch_loss",
        "reconstructed_acc",
        "node_success_count",
        "node_return_codes",
        "log_dir",
        "command",
    ]

    def _is_run_id(v):
        return isinstance(v, str) and v.startswith("run_")

    def _normalize_schema_if_needed(ws_local):
        """
        Ensure worksheet header matches current schema.
        Handles migration from the older 26-column schema and mixed rows.
        """
        current = [ws_local.cell(1, c).value for c in range(1, len(headers) + 1)]
        if current == headers:
            return

        # Detect legacy header (possibly with trailing None columns).
        legacy_detected = current[: len(old_headers)] == old_headers
        if not legacy_detected:
            # Unknown header layout: set canonical header for future writes.
            for c, h in enumerate(headers, start=1):
                ws_local.cell(1, c, h)
            return

        # Migrate existing rows to canonical layout.
        migrated_rows = []
        max_col = ws_local.max_column
        for r in range(2, ws_local.max_row + 1):
            row_vals = [ws_local.cell(r, c).value for c in range(1, max_col + 1)]
            if all(v is None for v in row_vals):
                continue

            row_dict_local = {}
            # Row written with old schema: col2 is run_id.
            if len(row_vals) >= 2 and _is_run_id(row_vals[1]):
                for idx, name in enumerate(old_headers):
                    col = idx + 1
                    row_dict_local[name] = row_vals[col - 1] if col - 1 < len(row_vals) else None
            # Row written with new schema while old header remained: col4 is run_id.
            elif len(row_vals) >= 4 and _is_run_id(row_vals[3]):
                for idx, name in enumerate(headers):
                    col = idx + 1
                    row_dict_local[name] = row_vals[col - 1] if col - 1 < len(row_vals) else None
            else:
                # Fallback: assume canonical positional mapping.
                for idx, name in enumerate(headers):
                    col = idx + 1
                    row_dict_local[name] = row_vals[col - 1] if col - 1 < len(row_vals) else None

            migrated_rows.append([row_dict_local.get(h, "") for h in headers])

        # Rewrite sheet with canonical header + migrated rows.
        ws_local.delete_rows(1, ws_local.max_row)
        ws_local.append(headers)
        for row in migrated_rows:
            ws_local.append(row)

    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    def _write_one(path_obj: Path):
        if path_obj.exists():
            wb_local = load_workbook(path_obj)
            ws_local = wb_local.active
        else:
            wb_local = Workbook()
            ws_local = wb_local.active
            ws_local.title = "runs"
        _normalize_schema_if_needed(ws_local)
        ws_local.append([row_dict.get(h, "") for h in headers])
        wb_local.save(path_obj)

    try:
        _write_one(xlsx_path)
        return str(xlsx_path)
    except PermissionError:
        # Common when the workbook is open in Excel/LibreOffice.
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = xlsx_path.with_name(f"{xlsx_path.stem}_{ts}{xlsx_path.suffix}")
        _write_one(fallback)
        return str(fallback)


def _run_headless_and_record(args):
    """Run all nodes in the current process and optionally export run summary to Excel."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = Path("logs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmdline_path = run_dir / "command.txt"
    cmdline_path.write_text(" ".join(sys.argv), encoding="utf-8")
    (run_dir / "params.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    procs = []
    log_paths = {}
    start_t = time.time()
    start_iso = datetime.datetime.now().isoformat(timespec="seconds")
    node_start_times = {}
    node_elapsed = {}
    node_ps = {}
    node_mem_baseline = {}
    node_mem_peak = {}
    for node_id in range(1, args.n_nodes + 1):
        cmd = _build_node_command(args, node_id)
        log_path = run_dir / f"node_{node_id}.log"
        log_paths[node_id] = log_path
        log_f = open(log_path, "w", encoding="utf-8")
        node_start_times[node_id] = time.time()
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        if psutil is not None:
            try:
                proc = psutil.Process(p.pid)
                node_ps[node_id] = proc
                rss = float(proc.memory_info().rss)
                node_mem_baseline[node_id] = rss
                node_mem_peak[node_id] = rss
            except Exception:
                pass
        procs.append((node_id, p, log_f, cmd))
        print(f"[headless] started node {node_id} -> {log_path}")
        time.sleep(1)

    client_proc = None
    client_log_f = None
    if args.start_client_distributor:
        client_cmd = [
            sys.executable,
            "-u",
            "client_distributor.py",
            "--client-node-id", str(args.dataset_source_node_id),
            "--n-nodes", str(args.n_nodes),
            "--t", str(args.t),
            "--base-port", str(args.base_port),
            "--host", str(args.host),
            "--mnist-samples", str(args.mnist_samples),
            "--client-test-samples", str(args.client_test_samples),
            "--field-size", str(args.field_size),
            "--scale-factor", str(args.scale_factor),
            "--seed", str(args.seed),
            "--barrier-timeout", str(args.dataset_distribution_timeout),
            "--membership-epoch", str(args.membership_epoch),
        ]
        if args.client_eval_after_training:
            client_cmd.extend([
                "--collect-client-eval",
                "--client-eval-samples", str(args.client_eval_samples),
                "--eval-timeout", str(args.client_eval_timeout),
            ])
        client_log_path = run_dir / "client_distributor.log"
        client_log_f = open(client_log_path, "w", encoding="utf-8")
        client_proc = subprocess.Popen(client_cmd, stdout=client_log_f, stderr=subprocess.STDOUT)
        print(f"[headless] started client distributor -> {client_log_path}")

    return_codes = {}
    try:
        finished = set()
        while len(finished) < len(procs):
            if client_proc is not None:
                client_rc = client_proc.poll()
                if client_rc is not None:
                    print(f"[headless] client distributor exited with code {client_rc}")
                    try:
                        if client_log_f is not None:
                            client_log_f.close()
                    except Exception:
                        pass
                    client_proc = None

            for node_id, p, log_f, _ in procs:
                if node_id in finished:
                    continue
                if psutil is not None:
                    proc = node_ps.get(node_id)
                    if proc is not None:
                        try:
                            rss = float(proc.memory_info().rss)
                            prev = node_mem_peak.get(node_id, rss)
                            if rss > prev:
                                node_mem_peak[node_id] = rss
                        except Exception:
                            pass
                rc = p.poll()
                if rc is None:
                    continue
                return_codes[node_id] = rc
                node_elapsed[node_id] = round(time.time() - node_start_times.get(node_id, start_t), 3)
                try:
                    log_f.close()
                except Exception:
                    pass
                finished.add(node_id)
                print(f"[headless] node {node_id} exited with code {rc}")
            if len(finished) < len(procs):
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping nodes...")
        for _, p, _, _ in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for _, p, _, _ in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        raise

    duration = round(time.time() - start_t, 2)
    end_iso = datetime.datetime.now().isoformat(timespec="seconds")
    if client_proc is not None:
        try:
            client_proc.wait(timeout=5)
        except Exception:
            try:
                client_proc.terminate()
            except Exception:
                pass
        try:
            if client_log_f is not None:
                client_log_f.close()
        except Exception:
            pass

    per_node_metrics = {}
    for node_id, log_path in log_paths.items():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            per_node_metrics[node_id] = _extract_metrics_from_log(text)
        except Exception:
            per_node_metrics[node_id] = {}

    client_metrics = {}
    if args.start_client_distributor:
        client_log_path = run_dir / "client_distributor.log"
        try:
            client_text = client_log_path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Client Final Accuracy \((\d+) samples\):\s*([0-9.]+)%", client_text)
            if m:
                client_metrics["client_eval_samples"] = int(m.group(1))
                client_metrics["client_final_accuracy_pct"] = float(m.group(2))
            m_dist = re.search(r"Client Distribution Time:\s*([0-9.]+)s", client_text)
            if m_dist:
                client_metrics["client_distribution_time_sec"] = float(m_dist.group(1))
            m_eval = re.search(r"Client Eval Time:\s*([0-9.]+)s", client_text)
            if m_eval:
                client_metrics["client_eval_time_sec"] = float(m_eval.group(1))
        except Exception:
            client_metrics = {}

    # Prefer node 1 for final report, fallback to first node with values.
    chosen = per_node_metrics.get(1, {})
    if "last_epoch_acc_pct" not in chosen:
        for m in per_node_metrics.values():
            if "last_epoch_acc_pct" in m:
                chosen = m
                break
    if "last_epoch_loss" not in chosen:
        for m in per_node_metrics.values():
            if "last_epoch_loss" in m:
                chosen = m
                break
    reconstructed_acc = None
    avg_batch_time_sec = None
    max_batch_time_sec = None
    prover_time_sec = None
    dataset_share_prep_time_sec = None
    training_time_sec = None
    client_eval_upload_time_sec = None
    for m in per_node_metrics.values():
        if "reconstructed_acc" in m:
            reconstructed_acc = m["reconstructed_acc"]
            break
    for m in per_node_metrics.values():
        if "avg_batch_time_sec" in m:
            avg_batch_time_sec = m["avg_batch_time_sec"]
            max_batch_time_sec = m.get("max_batch_time_sec")
            break
    logged_prover_times = [
        float(m["prover_time_sec_logged"])
        for m in per_node_metrics.values()
        if "prover_time_sec_logged" in m
    ]
    if logged_prover_times:
        prover_time_sec = max(logged_prover_times)
    dataset_times = [
        float(m["dataset_share_prep_time_sec"])
        for m in per_node_metrics.values()
        if "dataset_share_prep_time_sec" in m
    ]
    if dataset_times:
        dataset_share_prep_time_sec = max(dataset_times)
    training_times = [
        float(m["training_time_sec"])
        for m in per_node_metrics.values()
        if "training_time_sec" in m
    ]
    if training_times:
        training_time_sec = max(training_times)
    eval_upload_times = [
        float(m["client_eval_upload_time_sec"])
        for m in per_node_metrics.values()
        if "client_eval_upload_time_sec" in m
    ]
    if eval_upload_times:
        client_eval_upload_time_sec = max(eval_upload_times)
    if prover_time_sec is None and node_elapsed:
        prover_time_sec = round(float(node_elapsed.get(1, min(node_elapsed.values()))), 3)
    if "client_final_accuracy_pct" in client_metrics:
        reconstructed_acc = float(client_metrics["client_final_accuracy_pct"])

    # Memory summaries (MB)
    node_mem_overhead = {}
    for nid, base in node_mem_baseline.items():
        peak = node_mem_peak.get(nid, base)
        node_mem_overhead[nid] = max(0.0, peak - base)
    mb = 1024.0 * 1024.0
    total_base_mb = round(sum(node_mem_baseline.values()) / mb, 3) if node_mem_baseline else None
    total_peak_mb = round(sum(node_mem_peak.values()) / mb, 3) if node_mem_peak else None
    total_overhead_mb = round(sum(node_mem_overhead.values()) / mb, 3) if node_mem_overhead else None

    success_count = sum(1 for m in per_node_metrics.values() if m.get("status_ok"))
    all_zero = all(code == 0 for code in return_codes.values()) if return_codes else False
    status = "success" if all_zero else "partial_or_failed"

    print("\nRun summary:")
    print(f"  status: {status}")
    if "last_epoch_acc_pct" in chosen:
        print(f"  final_epoch_acc_pct: {chosen['last_epoch_acc_pct']}")
    if "last_epoch_loss" in chosen:
        print(f"  final_epoch_loss: {chosen['last_epoch_loss']}")
    if reconstructed_acc is not None:
        print(f"  reconstructed_acc: {reconstructed_acc}")
    print(f"  duration_sec: {duration}")
    if prover_time_sec is not None:
        print(f"  prover_time_sec: {prover_time_sec}")
    if dataset_share_prep_time_sec is not None:
        print(f"  dataset_share_prep_time_sec: {dataset_share_prep_time_sec}")
    if training_time_sec is not None:
        print(f"  training_time_sec: {training_time_sec}")
    if client_metrics.get("client_distribution_time_sec") is not None:
        print(f"  client_distribution_time_sec: {client_metrics.get('client_distribution_time_sec')}")
    if client_metrics.get("client_eval_time_sec") is not None:
        print(f"  client_eval_time_sec: {client_metrics.get('client_eval_time_sec')}")
    if client_eval_upload_time_sec is not None:
        print(f"  client_eval_upload_time_sec: {client_eval_upload_time_sec}")
    if total_base_mb is not None:
        print(f"  memory_mb: baseline={total_base_mb}, peak={total_peak_mb}, overhead={total_overhead_mb}")
    print(f"  logs: {run_dir}")

    if args.record_results_xlsx:
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_start": start_iso,
            "run_end": end_iso,
            "run_id": run_id,
            "status": status,
            "duration_sec": duration,
            "duration_min": round(duration / 60.0, 3),
            "n_nodes": args.n_nodes,
            "dataset": args.dataset,
            "batched": bool(args.batched),
<<<<<<< HEAD
            "train_mode": "secure",
=======
            "train_mode": args.train_mode,
>>>>>>> main
            "loss_mode": args.loss_mode,
            "softmax_grad_mode": args.softmax_grad_mode,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "field_size": args.field_size,
            "scale_factor": args.scale_factor,
            "softmax_temperature": args.softmax_temperature,
            "grad_clip": args.grad_clip,
            "logit_clip": args.logit_clip,
            "seed": args.seed,
            "final_epoch_acc_pct": chosen.get("last_epoch_acc_pct", ""),
            "final_epoch_loss": chosen.get("last_epoch_loss", ""),
            "reconstructed_acc": reconstructed_acc if reconstructed_acc is not None else "",
            "node_success_count": success_count,
            "node_elapsed_sec": json.dumps(node_elapsed),
            "prover_time_sec": prover_time_sec if prover_time_sec is not None else "",
            "avg_batch_time_sec": avg_batch_time_sec if avg_batch_time_sec is not None else "",
            "max_batch_time_sec": max_batch_time_sec if max_batch_time_sec is not None else "",
            "dataset_share_prep_time_sec": dataset_share_prep_time_sec if dataset_share_prep_time_sec is not None else "",
            "training_time_sec": training_time_sec if training_time_sec is not None else "",
            "client_distribution_time_sec": client_metrics.get("client_distribution_time_sec", ""),
            "client_eval_time_sec": client_metrics.get("client_eval_time_sec", ""),
            "client_eval_upload_time_sec": client_eval_upload_time_sec if client_eval_upload_time_sec is not None else "",
            "memory_baseline_mb": total_base_mb if total_base_mb is not None else "",
            "memory_peak_mb": total_peak_mb if total_peak_mb is not None else "",
            "memory_overhead_mb": total_overhead_mb if total_overhead_mb is not None else "",
            "node_memory_baseline_mb": json.dumps({k: round(v / mb, 3) for k, v in node_mem_baseline.items()}),
            "node_memory_peak_mb": json.dumps({k: round(v / mb, 3) for k, v in node_mem_peak.items()}),
            "node_memory_overhead_mb": json.dumps({k: round(v / mb, 3) for k, v in node_mem_overhead.items()}),
            "node_return_codes": json.dumps(return_codes),
            "log_dir": str(run_dir),
            "command": " ".join(sys.argv),
        }
        recorded_path = _append_run_to_xlsx(args.record_results_xlsx, row)
        print(f"  recorded_to_xlsx: {recorded_path}")
        if str(recorded_path) != str(args.record_results_xlsx):
            print("  note: primary xlsx path was not writable (likely open/locked); wrote fallback file instead.")


def main():
<<<<<<< HEAD
    args = parse_and_validate_args()
=======
    parser = argparse.ArgumentParser(description='Start all SENTRA nodes')
    parser.add_argument('--n-nodes', type=int, default=5,
                       help='Number of nodes to start (default: 5)')
    parser.add_argument('--base-port', type=int, default=8000,
                       help='Base port number (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host for all nodes (default: localhost)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Mini-batch size passed to each node (default: 8)')
    parser.add_argument('--accum-steps', type=int, default=1,
                       help='Gradient accumulation grouping for batched runner; effective batch = batch-size * accum-steps')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Epoch count passed to each node (default: 1)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate passed to each node (default: 0.01)')
    parser.add_argument('--t', type=int, default=1,
                       help='Privacy threshold (default: 1)')
    parser.add_argument('--s', type=int, default=1,
                       help='Adversarial share limit (default: 1)')
    parser.add_argument('--dataset', choices=['synthetic', 'mnist'], default='synthetic',
                       help='Dataset mode for each node (default: synthetic)')
    parser.add_argument('--mnist-samples', type=int, default=128,
                       help='MNIST sample count per node when --dataset mnist (default: 128)')
    parser.add_argument('--mnist-input-dim', type=int, default=64,
                       help='Flattened MNIST features to keep when --dataset mnist (default: 64, max: 784)')
    parser.add_argument('--mnist-hidden-dim', type=int, default=16,
                       help='Hidden layer width for MNIST when --dataset mnist (default: 16)')
    parser.add_argument('--mnist-test-samples', type=int, default=1000,
                       help='MNIST test sample count for post-training accuracy proxy (default: 1000)')
    parser.add_argument('--no-test-accuracy', action='store_true',
                       help='Disable post-training MNIST test accuracy reporting on nodes')
    parser.add_argument('--post-metrics-barrier-timeout', type=float, default=180.0,
                       help='Seconds to wait at post-metrics barrier before shutdown (default: 180)')
    parser.add_argument('--seed', type=int, default=2026,
                       help='Global random seed passed to all nodes (default: 2026)')
    parser.add_argument('--train-mode', choices=['secure', 'hybrid'], default='secure',
                       help='Training mode passed to all nodes (default: secure)')
    parser.add_argument('--batched', action='store_true',
                       help='Run the batched version of the MNIST MLP (much faster)')
    parser.add_argument('--loss-mode', choices=['mse', 'softmax'], default='softmax',
                       help='Output gradient mode for batched MNIST secure training (default: softmax)')
    parser.add_argument('--scale-factor', type=int, default=2**20,
                       help='Fixed-point scale for batched secure MNIST (default: 2^20 = 1048576)')
    parser.add_argument('--field-size', type=int, default=2**32 - 5,
                       help='Finite field modulus for batched secure MNIST (default: 2^32-5)')
    parser.add_argument('--softmax-temperature', type=float, default=1.0,
                       help='Softmax temperature for batched secure MNIST (default: 1.0)')
    parser.add_argument('--grad-clip', type=float, default=2.0,
                       help='Gradient clip bound for batched secure MNIST (default: 2.0)')
    parser.add_argument('--logit-clip', type=float, default=8.0,
                       help='Logit clip bound before secure exp approximation (default: 8.0)')
    parser.add_argument('--exp-approx', choices=['taylor5', 'pade22'], default='pade22',
                       help='Secure exp approximation used in softmax (default: pade22)')
    parser.add_argument('--softmax-grad-mode', choices=['secure_approx', 'opened_exact'], default='secure_approx',
                       help='Softmax CE gradient path in batched mode (default: secure_approx)')
    parser.add_argument('--explode-logit-threshold', type=float, default=10.0,
                       help='Logit explosion threshold for batched secure MNIST (default: 10.0)')
    parser.add_argument('--loss-growth-threshold', type=float, default=5.0,
                       help='Loss growth threshold for instability checks (default: 5.0)')
    parser.add_argument('--grad-norm-threshold', type=float, default=5.0,
                       help='Estimated gradient norm threshold for instability checks (default: 5.0)')
    parser.add_argument('--no-abort-on-instability', action='store_true',
                       help='Do not abort batched secure training when instability is detected')
    parser.add_argument('--debug-numerics', action='store_true',
                       help='Enable numeric probes (updates/logits/dz2) in batched secure mode')
    parser.add_argument('--debug-division', action='store_true',
                       help='Enable secure division debug summaries in batched secure mode')
    parser.add_argument('--packed-forward-pilot', action='store_true',
                       help='Enable packed forward kernel pilot mode in batched secure runner')
    parser.add_argument('--packed-forward-native', action='store_true',
                       help='Enable experimental packed-native dense1 forward kernel in batched secure runner')
    parser.add_argument('--packed-end2end', action='store_true',
                       help='Enable experimental packed API path in batched secure runner')
    parser.add_argument('--dpss-refresh-interval', type=int, default=0,
                       help='Herzberg proactive refresh every N epochs in batched mode (0=disabled). Requires --batched --enable-network.')
    parser.add_argument('--membership-epoch', type=int, default=0,
                       help='Membership epoch e for batched nodes + client distributor: MPC/barrier prefix m{e}_ (default: 0).')
    parser.add_argument('--enable-failure-detection', action='store_true',
                        help='Enable heartbeat-based failure detection; use dynamic n_active for packing safety.')
    parser.add_argument(
        '--enable-dropout-reshare-recovery',
        action='store_true',
        help='Forward to batched runner: on failure, Lagrange reshare weights among survivors and resume if safe.',
    )
    parser.add_argument(
        '--enable-join-recovery',
        action='store_true',
        help='Forward to batched runner: on node rejoin, Lagrange reshare weights to new committee and resume.',
    )
    parser.add_argument('--export-reconstructed-model', type=str, default='',
                       help='Export reconstructed final model to this .npz path (opener node writes file)')
    parser.add_argument('--export-timeout', type=float, default=180.0,
                       help='Timeout in seconds for final model reconstruction/export (default: 180)')
    parser.add_argument('--distribute-dataset-shares', action='store_true',
                       help='Owner node shares MNIST to peers (local simulation mode).')
    parser.add_argument('--dataset-owner-node', type=int, default=1,
                       help='Owner node id for --distribute-dataset-shares (default: 1).')
    parser.add_argument('--receive-dataset-shares-from-client', action='store_true',
                       help='Nodes receive only pre-shared dataset from external client distributor.')
    parser.add_argument('--dataset-source-node-id', type=int, default=0,
                       help='External sender node_id for client distributor mode (default: 0).')
    parser.add_argument('--dataset-distribution-timeout', type=float, default=900.0,
                       help='Timeout for dataset share distribution/reception barrier (seconds).')
    parser.add_argument('--start-client-distributor', action='store_true',
                       help='In headless mode, auto-start client_distributor.py after launching nodes.')
    parser.add_argument('--client-eval-after-training', action='store_true',
                       help='After training, nodes send final inference shares to client; client reconstructs accuracy.')
    parser.add_argument('--client-eval-samples', type=int, default=100,
                       help='Sample count for client-side reconstructed final accuracy.')
    parser.add_argument('--client-test-samples', type=int, default=-1,
                       help='Number of test shares to distribute from client (-1: auto; with client eval uses client-eval-samples).')
    parser.add_argument('--client-eval-timeout', type=float, default=0.0,
                       help='Timeout for client-side evaluation share collection and barrier in seconds (0 = no timeout).')
    parser.add_argument('--use-kvs-dataset', action='store_true',
                       help='Store dataset shares in local KVS and retrieve mini-batches from KVS. Requires distributed dataset.')
    parser.add_argument('--use-weight-versioning', action='store_true',
                       help='Store weight shares to local KVS with v_theta after each epoch.')
    parser.add_argument('--headless', action='store_true',
                       help='Run all nodes in this terminal and wait for completion (writes per-node logs)')
    parser.add_argument('--record-results-xlsx', type=str, default='',
                       help='Append run parameters + parsed final metrics to an Excel file (implies --headless)')
    
    args = parser.parse_args()
    
    if args.batched:
        args.dataset = 'mnist'

    if args.distribute_dataset_shares and args.receive_dataset_shares_from_client:
        raise ValueError("Use only one dataset-sharing mode: owner-node or external client distributor")
    if args.start_client_distributor and not args.receive_dataset_shares_from_client:
        raise ValueError("--start-client-distributor requires --receive-dataset-shares-from-client")
    if args.client_eval_after_training and not args.receive_dataset_shares_from_client:
        raise ValueError("--client-eval-after-training requires --receive-dataset-shares-from-client")
    if getattr(args, "enable_dropout_reshare_recovery", False):
        if not bool(args.enable_failure_detection):
            raise ValueError("--enable-dropout-reshare-recovery requires --enable-failure-detection")
        if not bool(args.batched):
            raise ValueError("--enable-dropout-reshare-recovery requires --batched")
    if getattr(args, "enable_join_recovery", False):
        if not bool(args.enable_failure_detection):
            raise ValueError("--enable-join-recovery requires --enable-failure-detection")
        if not bool(args.batched):
            raise ValueError("--enable-join-recovery requires --batched")
    if getattr(args, "use_kvs_dataset", False):
        if not (bool(args.distribute_dataset_shares) or bool(args.receive_dataset_shares_from_client)):
            raise ValueError("--use-kvs-dataset requires --distribute-dataset-shares or --receive-dataset-shares-from-client")
    if (
        bool(args.packed_forward_native)
        and bool(args.start_client_distributor)
        and bool(args.client_eval_after_training)
        and float(args.client_eval_timeout) > 0.0
    ):
        print(
            "Packed native forward is enabled; overriding --client-eval-timeout to 0 "
            "(no timeout) to avoid premature client disconnect during long runs."
        )
        args.client_eval_timeout = 0.0
>>>>>>> main
        
    print("=" * 70)
    print("Starting SENTRA Multi-Node Training")
    print("=" * 70)
    print(f"Total nodes: {args.n_nodes}")
    print(f"Base port: {args.base_port}")
    print(f"Host: {args.host}")
    print(f"Training: epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    if args.batched:
        print(f"Batched secure config: field={args.field_size}, scale={args.scale_factor}, temp={args.softmax_temperature}, grad_clip={args.grad_clip}, logit_clip={args.logit_clip}, exp={args.exp_approx}, grad_mode={args.softmax_grad_mode}, loss_mode={args.loss_mode}")
        print(f"Batch config: batch_size={args.batch_size}, accum_steps={args.accum_steps}, effective_batch={args.batch_size * args.accum_steps}")
        if args.distribute_dataset_shares:
            print(f"Dataset sharing mode: owner-node (owner={args.dataset_owner_node})")
        elif args.receive_dataset_shares_from_client:
            print(f"Dataset sharing mode: external client sender (source_node_id={args.dataset_source_node_id})")
        if args.client_eval_after_training:
            print(f"Client-side eval after training: enabled ({args.client_eval_samples} samples)")
        print(f"Membership epoch: e={args.membership_epoch} (prefix m{args.membership_epoch}_)")
        print(f"Instability thresholds: logit={args.explode_logit_threshold}, loss_growth={args.loss_growth_threshold}, grad_norm={args.grad_norm_threshold}, abort={not args.no_abort_on_instability}")
        print(
            f"Debug flags: numerics={args.debug_numerics}, division={args.debug_division}, "
            f"packed_forward_pilot={args.packed_forward_pilot}, "
            f"packed_forward_native={args.packed_forward_native}, "
            f"packed_end2end={args.packed_end2end}"
        )
<<<<<<< HEAD
    print(f"Thresholds: t={args.t}")
=======
    print(f"Thresholds: t={args.t}, s={args.s}")
>>>>>>> main
    print(f"Dataset: {args.dataset}")
    print("=" * 70)
    print()

    if args.record_results_xlsx and not args.headless:
        print("Enabling --headless because --record-results-xlsx was provided.")
        args.headless = True

    if args.headless:
        _run_headless_and_record(args)
        return

    processes = []
    
    for node_id in range(1, args.n_nodes + 1):
        cmd = _build_node_command(args, node_id)
        
        print(f"Starting Node {node_id}...")
        
        # Check if we are running under WSL or pure Linux/Windows
        import platform
        import os
        is_wsl = 'microsoft' in platform.release().lower() or 'wsl' in platform.release().lower()
        
        if sys.platform == 'win32':
            # Run in a new command prompt window on Windows
            subprocess.Popen(['start', 'cmd', '/k'] + cmd, shell=True)
        elif is_wsl:
            # Under WSL, we can call out to cmd.exe to launch a new wsl.exe window
            # Pass command to bash and append a read prompt so the window stays open
            cmd_str = " ".join(f"'{arg}'" if ' ' in arg else arg for arg in cmd)
            bash_cmd = f"{cmd_str}; echo ''; read -p 'Process complete. Press enter to close...'"
            wsl_cmd = ['cmd.exe', '/c', 'start', 'wsl.exe', '-e', 'bash', '-c', bash_cmd]
            subprocess.Popen(wsl_cmd)
        else:
            # Fallback for pure Linux (e.g. gnome-terminal, xterm). If none exists, run in background
            try:
                subprocess.Popen(['xterm', '-e'] + cmd)
            except Exception:
                # If xterm fails, fall back to logs
                f = open(f"node_{node_id}.log", "w")
                subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        
        time.sleep(1)  # Stagger starts
    
    print()
    print("All nodes started in separate windows.")
    print("Close each window to stop the corresponding node, or watch the live outputs!")
    print()
    print("To stop all nodes, close each window or press Ctrl+C.")


if __name__ == '__main__':
    main()
