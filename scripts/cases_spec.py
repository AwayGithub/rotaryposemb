#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RotaryPosEmb local PV cases — aligned with CANNJudge problem constraints.

Official constraints (docs/problem.md):
  x: (B, S, H, rope_dim), float16 | float32
  positions: (B, S), int32, values in [0, 131072]
  theta: float (default 10000.0)
  B in [1, 128], S in [1, 8192], H in [1, 256]
  rope_dim in [16, 1024] and even
  precision: fp32 rtol/atol < 1e-4; fp16 rtol/atol < 1e-3
  golden: FP32 inv_freq + cos/sin, then cast back (see RotaryPosEmb.impl)

Official examples covered:
  - (1, 8, 1, 16) fp32
  - (2, 512, 128, 64) fp16  DeepSeek-like
  - (2, 16, 3, 24) fp16    non-aligned rope_dim
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# dtype_code: 0=float32, 1=float16
# pos_spec:
#   ("arange",)           -> 0..S-1 clipped to 131072
#   ("zero",)             -> all zeros (y should == x)
#   ("uniform", lo, hi)   -> randint [lo, hi)

CaseSpec = Dict[str, Any]

CASES: List[CaseSpec] = [
    # 0-2: tiny / official example1
    {"id": 0, "dtype": 0, "shape": (1, 8, 1, 16), "theta": 10000.0, "pos": ("arange",), "tag": "official-ex1-fp32"},
    {"id": 1, "dtype": 0, "shape": (1, 7, 1, 16), "theta": 10000.0, "pos": ("uniform", 0, 50), "tag": "small-fp32"},
    {"id": 2, "dtype": 1, "shape": (1, 8, 1, 16), "theta": 10000.0, "pos": ("uniform", 0, 50), "tag": "small-fp16"},
    # 3-5: non-aligned rope_dim / H (official ex3 family)
    {"id": 3, "dtype": 1, "shape": (2, 16, 3, 24), "theta": 10000.0, "pos": ("uniform", 0, 8192), "tag": "official-ex3-dim24"},
    {"id": 4, "dtype": 1, "shape": (2, 7, 5, 18), "theta": 10000.0, "pos": ("uniform", 0, 8192), "tag": "unaligned-dim18"},
    {"id": 5, "dtype": 0, "shape": (3, 11, 2, 20), "theta": 10000.0, "pos": ("uniform", 0, 8192), "tag": "unaligned-dim20-fp32"},
    # 6-7: positions == 0  => y == x
    {"id": 6, "dtype": 1, "shape": (1, 9, 2, 16), "theta": 10000.0, "pos": ("zero",), "tag": "pos0-fp16"},
    {"id": 7, "dtype": 0, "shape": (1, 9, 2, 16), "theta": 10000.0, "pos": ("zero",), "tag": "pos0-fp32"},
    # 8-10: medium / DeepSeek-like heads
    {"id": 8, "dtype": 1, "shape": (2, 64, 8, 64), "theta": 10000.0, "pos": ("uniform", 0, 131072), "tag": "mid-fp16-wide-pos"},
    {"id": 9, "dtype": 0, "shape": (2, 64, 8, 64), "theta": 10000.0, "pos": ("uniform", 0, 131072), "tag": "mid-fp32-wide-pos"},
    {"id": 10, "dtype": 1, "shape": (2, 128, 32, 64), "theta": 10000.0, "pos": ("uniform", 0, 8192), "tag": "mid-fp16-h32"},
    # 11-14: theta variations
    {"id": 11, "dtype": 0, "shape": (1, 64, 4, 32), "theta": 1.0, "pos": ("uniform", 0, 1000), "tag": "theta-1"},
    {"id": 12, "dtype": 1, "shape": (2, 32, 8, 64), "theta": 1000.0, "pos": ("uniform", 0, 131072), "tag": "theta-1e3"},
    {"id": 13, "dtype": 1, "shape": (2, 32, 8, 64), "theta": 500000.0, "pos": ("uniform", 0, 131072), "tag": "theta-5e5"},
    {"id": 14, "dtype": 1, "shape": (2, 32, 8, 64), "theta": 3.5e4, "pos": ("uniform", 0, 131072), "tag": "theta-3.5e4"},
    # 15-16: larger dim edges (still within [16,1024])
    {"id": 15, "dtype": 0, "shape": (1, 128, 4, 256), "theta": 10000.0, "pos": ("uniform", 0, 131072), "tag": "dim256-fp32"},
    {"id": 16, "dtype": 0, "shape": (1, 16, 1, 1024), "theta": 10000.0, "pos": ("uniform", 0, 131072), "tag": "dim1024-edge"},
    # 17: official example2 DeepSeek V3-like
    {"id": 17, "dtype": 1, "shape": (2, 512, 128, 64), "theta": 10000.0, "pos": ("uniform", 0, 8192), "tag": "official-ex2-deepseek"},
    # 18-19: long seq / single token
    {"id": 18, "dtype": 1, "shape": (1, 4096, 1, 64), "theta": 10000.0, "pos": ("uniform", 0, 131072), "tag": "long-seq"},
    {"id": 19, "dtype": 0, "shape": (1, 1, 1, 16), "theta": 10000.0, "pos": ("uniform", 0, 50), "tag": "single-token"},
]


def np_dtype(dtype_code: int):
    return np.float32 if int(dtype_code) == 0 else np.float16


def precision_of(dtype_code: int) -> Tuple[float, float, float]:
    """rtol, atol, tol_frac — match official judge thresholds."""
    if int(dtype_code) == 0:
        return 1e-4, 1e-4, 0.0
    return 1e-3, 1e-3, 0.0


def validate_case(spec: CaseSpec) -> None:
    B, S, H, D = spec["shape"]
    assert 1 <= B <= 128, spec
    assert 1 <= S <= 8192, spec
    assert 1 <= H <= 256, spec
    assert 16 <= D <= 1024 and D % 2 == 0, spec
    assert int(spec["dtype"]) in (0, 1), spec


def make_positions(spec: CaseSpec, rng: np.random.Generator) -> np.ndarray:
    B, S, _, _ = spec["shape"]
    mode = spec["pos"]
    kind = mode[0]
    if kind == "zero":
        return np.zeros((B, S), dtype=np.int32)
    if kind == "arange":
        row = np.arange(S, dtype=np.int32) % 131073
        return np.broadcast_to(row, (B, S)).copy()
    if kind == "uniform":
        lo, hi = int(mode[1]), int(mode[2])
        hi = min(hi, 131073)
        lo = max(lo, 0)
        if hi <= lo:
            hi = lo + 1
        return rng.integers(lo, hi, size=(B, S), dtype=np.int32)
    raise ValueError(f"unknown pos mode: {mode}")


def make_x(spec: CaseSpec, rng: np.random.Generator) -> np.ndarray:
    dt = np_dtype(spec["dtype"])
    x = rng.standard_normal(spec["shape"]).astype(np.float32)
    # moderate dynamic range; avoid extreme overflow in fp16 path
    scale = float(10.0 ** rng.uniform(-0.5, 0.5))
    x = (x * scale).astype(dt)
    return x


for _c in CASES:
    validate_case(_c)

assert len(CASES) == 20, len(CASES)
assert [c["id"] for c in CASES] == list(range(20))
