#!/usr/bin/env python3
"""Ascend C Toolkit entry: gen 20 cases, run kernel, verify (problem-aligned).

Toolkit typically runs:  cd build && python ../run.py
Optional: python run.py [case_id|all]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BIN = BUILD / "rotary_pos_emb_custom"
SCRIPTS = ROOT / "scripts"
CORE_NUM = "0"


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_cann_env() -> None:
    if os.environ.get("ASCEND_HOME_PATH") and "libregister" in (
        os.environ.get("LD_LIBRARY_PATH") or ""
    ):
        return
    candidates = []
    if os.environ.get("ASCEND_HOME_PATH"):
        candidates.append(Path(os.environ["ASCEND_HOME_PATH"]) / "set_env.sh")
    candidates.extend(
        [
            Path("/home/developer/Ascend/cann-9.0.0/set_env.sh"),
            Path("/home/developer/Ascend/ascend-toolkit/set_env.sh"),
            Path("/home/developer/Ascend/cann/set_env.sh"),
        ]
    )
    for script in candidates:
        if not script.is_file():
            continue
        cmd = (
            f'set +u; source "{script}"; '
            r'python3 -c "import os,json; print(json.dumps(dict(os.environ)))"'
        )
        try:
            out = subprocess.check_output(["bash", "-lc", cmd], text=True)
            env = json.loads(out.strip().splitlines()[-1])
            os.environ.clear()
            os.environ.update(env)
            return
        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, ValueError):
            continue


def load_meta(case_id: int) -> dict:
    meta_path = BUILD / "input" / f"case{case_id}" / "meta.json"
    if not meta_path.is_file():
        die(f"missing {meta_path}; run gen_data first")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def run_one(case_id: int) -> bool:
    meta = load_meta(case_id)
    in_dir = BUILD / "input" / f"case{case_id}"
    out_dir = BUILD / "output" / f"case{case_id}"
    gold_dir = BUILD / "output" / f"golden_case{case_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_bin = in_dir / "x.bin"
    p_bin = in_dir / "positions.bin"
    y_bin = out_dir / "y.bin"
    if not x_bin.is_file() or not p_bin.is_file():
        die(f"case{case_id}: missing input bins")

    # flat copies for verify_result fallback / tooling
    flat = BUILD / "output"
    gold_y = gold_dir / "golden_y.bin"
    if gold_y.is_file():
        (flat / "golden_y.bin").write_bytes(gold_y.read_bytes())

    dtype = str(meta["dtype"])
    B, S, H, D = str(meta["B"]), str(meta["S"]), str(meta["H"]), str(meta["D"])
    theta = str(meta["theta"])

    cmd = [
        str(BIN),
        dtype, B, S, H, D, theta,
        str(x_bin), str(p_bin), str(y_bin), CORE_NUM,
    ]
    tag = meta.get("tag", "")
    print(
        f"--- case{case_id:02d} {tag} dtype={meta.get('dtype_name')} "
        f"shape=({B},{S},{H},{D}) theta={theta} ---"
    )
    t0 = time.time()
    try:
        subprocess.check_call(cmd, cwd=str(BUILD))
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] case{case_id} kernel exit {e.returncode}")
        return False
    ms = (time.time() - t0) * 1000.0
    print(f"  kernel wall {ms:.1f} ms")

    # also place y.bin flat for verify fallback
    if y_bin.is_file():
        (flat / "y.bin").write_bytes(y_bin.read_bytes())

    rc = subprocess.call(
        [sys.executable, str(SCRIPTS / "verify_result.py"), str(case_id)],
        cwd=str(BUILD),
    )
    ok = rc == 0
    print(f"  verify: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str]) -> int:
    if not BIN.is_file():
        die(f"binary not found: {BIN}\nRun Toolkit「编译部署算子」or: bash build.sh")

    ensure_cann_env()
    BUILD.mkdir(parents=True, exist_ok=True)
    os.chdir(BUILD)

    # which cases
    if len(argv) >= 2 and argv[1] not in ("all", "*"):
        try:
            case_ids = [int(argv[1])]
        except ValueError:
            die(f"usage: run.py [case_id|all]")
    else:
        case_ids = list(range(20))

    print("=== [1/3] Gen test data (20 cases) ===")
    subprocess.check_call([sys.executable, str(SCRIPTS / "gen_data.py")], cwd=str(BUILD))

    print(f"=== [2/3] Run kernel on NPU ({len(case_ids)} cases) ===")
    results = []
    for cid in case_ids:
        ok = run_one(cid)
        results.append((cid, ok))

    print("=== [3/3] Summary ===")
    n_pass = sum(1 for _, ok in results if ok)
    for cid, ok in results:
        print(f"  case{cid:02d}: {'PASS' if ok else 'FAIL'}")
    print(f"=== {n_pass}/{len(results)} PASSED ===")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except subprocess.CalledProcessError as e:
        die(f"command failed with exit {e.returncode}: {e.cmd}")
