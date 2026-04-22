"""Resolve CUDA library path mismatches before any deep-learning imports.

Spark ships CUDA 13. vLLM 0.x prebuilts link against libcudart.so.12.
Ollama bundles a libcudart.so.12 we can borrow.

The dynamic loader caches its library search path at process start, so
mutating `os.environ['LD_LIBRARY_PATH']` mid-run is too late. We have to
re-exec the Python process with the env set BEFORE python starts.

Call `ensure_cuda12_runtime()` BEFORE importing torch/vllm. If a re-exec is
needed, this function does NOT return — execvpe replaces the process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CANDIDATE_PATHS = [
    Path("/usr/local/lib/ollama/cuda_v12"),
    Path("/usr/local/cuda-12.0/lib64"),
    Path("/usr/local/cuda-12.1/lib64"),
    Path("/usr/local/cuda-12.4/lib64"),
]

_REEXEC_FLAG = "_PYLOX_CUDA_REEXEC_DONE"


def ensure_cuda12_runtime() -> None:
    """If libcudart.so.12 paths aren't already in LD_LIBRARY_PATH, re-exec with them."""
    if os.environ.get(_REEXEC_FLAG):
        return         # already re-exec'd, don't loop

    extras = [str(p) for p in CANDIDATE_PATHS if (p / "libcudart.so.12").exists()]
    if not extras:
        return         # no CUDA 12 runtime available, nothing we can do

    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if all(e in cur for e in extras):
        return         # already set

    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = ":".join(extras + ([cur] if cur else []))
    new_env[_REEXEC_FLAG] = "1"

    # Re-exec the same Python interpreter + script + args
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)


# Auto-run on import — anyone who imports this module gets it transparently.
ensure_cuda12_runtime()
