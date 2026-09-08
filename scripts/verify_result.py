#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify kernel output vs golden for one case (official rtol/atol)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cases_spec import CASES, np_dtype, precision_of

# Build per-case specs: (name, dtype, rtol, atol, tol_frac)
case_output_specs = {}
for spec in CASES:
    cid = int(spec["id"])
    rtol, atol, tol = precision_of(spec["dtype"])
    case_output_specs[cid] = [
        ("y", np_dtype(spec["dtype"]), rtol, atol, tol),
    ]


def verify_result(output_path, golden_path, dtype, rtol, atol, tol=0.0):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    total_size = golden.size

    if output.size != golden.size:
        print(
            f"FAILED: output has {output.size} elements, "
            f"golden has {total_size} — size mismatch"
        )
        return False

    cmp_output = output.astype(np.float32)
    cmp_golden = golden.astype(np.float32)
    isclose = np.isclose(cmp_output, cmp_golden, rtol=rtol, atol=atol, equal_nan=True)
    errors = int(np.sum(~isclose))
    error_rate = errors / total_size if total_size > 0 else 0.0
    diff = np.abs(cmp_output - cmp_golden)
    max_diff = float(np.max(diff)) if diff.size else 0.0

    if error_rate <= tol:
        print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        if errors > 0:
            print(f"  Mismatched: {errors}/{total_size} ({error_rate * 100:.4f}%), tol={tol}")
        print(f"  Max diff: {max_diff} (rtol={rtol}, atol={atol})")
        return True

    print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
    print(f"  Mismatched: {errors}/{total_size} ({error_rate * 100:.4f}%), tol={tol}")
    print(f"  Max diff: {max_diff} (rtol={rtol}, atol={atol})")
    # show a few bad indices
    bad = np.where(~isclose)[0][:5]
    for i in bad:
        print(f"  idx[{i}] out={cmp_output[i]} gold={cmp_golden[i]} diff={diff[i]}")
    return False


def main() -> int:
    try:
        case_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    except ValueError:
        print(f"Invalid case_id: {sys.argv[1] if len(sys.argv) > 1 else '(none)'}")
        return 1

    if case_id not in case_output_specs:
        print(f"Unknown case_id {case_id}. Available: {sorted(case_output_specs.keys())}")
        return 1

    # Prefer per-case dirs; fall back to flat layout for case0 compatibility
    out_case = os.path.join("output", f"case{case_id}")
    gold_case = os.path.join("output", f"golden_case{case_id}")
    flat_out = "output"

    all_pass = True
    for name, dtype, rtol, atol, tol in case_output_specs[case_id]:
        candidates = [
            (
                os.path.join(out_case, f"{name}.bin"),
                os.path.join(gold_case, f"golden_{name}.bin"),
            ),
            (
                os.path.join(flat_out, f"{name}.bin"),
                os.path.join(flat_out, f"golden_{name}.bin"),
            ),
            (
                os.path.join(flat_out, f"{name}.bin"),
                os.path.join(gold_case, f"golden_{name}.bin"),
            ),
        ]
        pair = None
        for op, gp in candidates:
            if os.path.exists(op) and os.path.exists(gp):
                pair = (op, gp)
                break
        if pair is None:
            print(f"FAILED: case{case_id} missing {name}.bin or golden_{name}.bin")
            print(f"  looked under {out_case}/ and {flat_out}/")
            all_pass = False
            continue
        if not verify_result(pair[0], pair[1], dtype, rtol, atol, tol):
            all_pass = False
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
