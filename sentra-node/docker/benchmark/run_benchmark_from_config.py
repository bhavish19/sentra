#!/usr/bin/env python3
"""
Load a YAML benchmark profile and invoke start_all_nodes (headless multi-node MNIST).

YAML format: flat mapping of CLI-style keys (hyphens), e.g.:
  headless: true
  n-nodes: 3
  num-epochs: 1
Booleans: true -> flag present, false -> omitted.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _resolve_node_root() -> Path:
    env = (os.environ.get("SENTRA_NODE_ROOT") or "").strip()
    if env:
        return Path(env)
    p = Path("/workspace/node")
    if p.is_dir():
        return p
    # Repo checkout: .../sentra-node/docker/benchmark/this_file.py
    here = Path(__file__).resolve()
    candidate = here.parents[2] / "python" / "node"
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        "Could not resolve node root. Set SENTRA_NODE_ROOT to sentra-node/python/node "
        "or run inside the benchmark Docker image."
    )


_NODE_ROOT = _resolve_node_root()
if str(_NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_NODE_ROOT))


def _yaml_value_to_argv(flag: str, value: Any) -> List[str]:
    if isinstance(value, bool):
        return [flag] if value else []
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for v in value:
            parts.extend([flag, str(v)])
        return parts
    return [flag, str(value)]


def config_to_argv(data: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, value in sorted(data.items()):
        if key.startswith("_") or key in ("comment", "comments"):
            continue
        if isinstance(value, dict):
            raise SystemExit(f"Unsupported nested mapping at key {key!r}; use flat CLI-style keys.")
        flag = "--" + str(key).replace("_", "-")
        argv.extend(_yaml_value_to_argv(flag, value))
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SENTRA benchmark from YAML config")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML file (see /workspace/benchmark-configs in the image)",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional args after -- are passed through to start_all_nodes.py",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SystemExit("YAML root must be a mapping")

    argv = config_to_argv(raw)
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    argv.extend(extra)

    import start_all_nodes

    old = sys.argv
    try:
        sys.argv = [str(Path(start_all_nodes.__file__).resolve())] + argv
        start_all_nodes.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
