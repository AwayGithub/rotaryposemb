#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate 20 PV cases + golden aligned with official RotaryPosEmb constraints."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from RotaryPosEmb import impl
from cases_spec import CASES, make_positions, make_x, np_dtype

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

SEED = 20260908


def main() -> None:
    for spec in CASES:
        cid = int(spec["id"])
        rng = np.random.default_rng(SEED + cid * 997)
        shape = tuple(spec["shape"])
        B, S, H, D = shape
        dtype_code = int(spec["dtype"])
        theta = float(spec["theta"])
        dt = np_dtype(dtype_code)

        x = make_x(spec, rng)
        positions = make_positions(spec, rng)
        assert x.shape == shape and x.dtype == dt
        assert positions.shape == (B, S) and positions.dtype == np.int32
        assert positions.min() >= 0 and positions.max() <= 131072

        golden = impl(x, positions, theta=theta)
        assert golden is not None and golden.shape == shape

        in_dir = os.path.join("input", f"case{cid}")
        g_dir = os.path.join("output", f"golden_case{cid}")
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(g_dir, exist_ok=True)

        x.tofile(os.path.join(in_dir, "x.bin"))
        positions.tofile(os.path.join(in_dir, "positions.bin"))
        golden.tofile(os.path.join(g_dir, "golden_y.bin"))

        meta = {
            "id": cid,
            "tag": spec.get("tag", ""),
            "dtype": dtype_code,
            "dtype_name": "float32" if dtype_code == 0 else "float16",
            "B": B,
            "S": S,
            "H": H,
            "D": D,
            "theta": theta,
            "pos": list(spec["pos"]),
            "x_nbytes": int(x.nbytes),
            "p_nbytes": int(positions.nbytes),
            "y_nbytes": int(golden.nbytes),
        }
        with open(os.path.join(in_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(
            f"[gen] case{cid:02d} {spec.get('tag','')} "
            f"dtype={meta['dtype_name']} shape={shape} theta={theta} "
            f"pos=[{positions.min()},{positions.max()}]"
        )

    print(f"Generated test data and golden output for {len(CASES)} cases.")


if __name__ == "__main__":
    main()
