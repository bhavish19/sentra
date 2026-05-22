"""Local MNIST loading (no Keras / network download)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

_DOCKER_PATH = Path("/mnist.npz")
_KERAS_CACHE_PATH = Path.home() / ".keras" / "datasets" / "mnist.npz"


def _repo_resource_paths() -> list[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "demonstrator" / "backend" / "resources" / "mnist.npz"
        if candidate.is_file():
            return [candidate]
    return []


def resolve_mnist_npz_path() -> Path:
    override = os.environ.get("MNIST_NPZ_PATH", "").strip()
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"MNIST_NPZ_PATH is set but file not found: {p}")
        return p

    for p in (_DOCKER_PATH, _KERAS_CACHE_PATH, *_repo_resource_paths()):
        if p.is_file():
            return p

    searched = ", ".join(str(c) for c in (_DOCKER_PATH, _KERAS_CACHE_PATH, *_repo_resource_paths()))
    raise FileNotFoundError(
        "Local MNIST not found (downloads are disabled). "
        "Use Docker (mnist.npz baked at /mnist.npz), set MNIST_NPZ_PATH, "
        "copy demonstrator/backend/resources/mnist.npz to ~/.keras/datasets/mnist.npz, "
        f"or place it under demonstrator/backend/resources/. Searched: {searched}"
    )


def loadMNISTDataset() -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """Load MNIST train/test (uint8, 28x28) from local mnist.npz only."""
    path = resolve_mnist_npz_path()
    data = np.load(path)
    x_train = np.asarray(data["x_train"])
    y_train = np.asarray(data["y_train"])
    x_test = np.asarray(data["x_test"])
    y_test = np.asarray(data["y_test"])
    return (x_train, y_train), (x_test, y_test)
