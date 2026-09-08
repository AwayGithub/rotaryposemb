#!/usr/bin/env python3
"""Ascend C Toolkit entry: generate data, run kernel, verify case0.

Toolkit typically runs:  cd build && python ../run.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BIN = BUILD / "rotary_pos_emb_custom"
SCRIPTS = ROOT / "scripts"

# case0 from scripts/gen_data.py: fp32 (1,7,1,16) theta=10000
DTYPE = "0"
B, S, H, D = "1", "7", "1", "16"
THETA = "10000.0"
CORE_NUM = "0"


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ensure_cann_env() -> None:
    """Source CANN set_env into current process env if libs missing."""
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
        # Export env from a login-like bash after sourcing set_env.sh
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


def main() -> int:
    if not BIN.is_file():
        die(f"binary not found: {BIN}\nRun Toolkit「编译部署算子」or: bash build.sh")

    ensure_cann_env()

    BUILD.mkdir(parents=True, exist_ok=True)
    os.chdir(BUILD)

    print("=== [1/3] Gen test data ===")
    subprocess.check_call([sys.executable, str(SCRIPTS / "gen_data.py")], cwd=str(BUILD))

    case0 = BUILD / "input" / "case0"
    inp = BUILD / "input"
    out = BUILD / "output"
    out.mkdir(parents=True, exist_ok=True)
    if case0.is_dir():
        for name in ("x.bin", "positions.bin"):
            src = case0 / name
            dst = inp / name
            if src.is_file():
                dst.write_bytes(src.read_bytes())
    golden_src = out / "golden_case0" / "golden_y.bin"
    golden_dst = out / "golden_y.bin"
    if golden_src.is_file():
        golden_dst.write_bytes(golden_src.read_bytes())

    x_bin = inp / "x.bin"
    p_bin = inp / "positions.bin"
    y_bin = out / "y.bin"
    if not x_bin.is_file() or not p_bin.is_file():
        die(f"missing input bins under {inp}")

    print("=== [2/3] Run kernel on NPU ===")
    cmd = [
        str(BIN),
        DTYPE, B, S, H, D, THETA,
        str(x_bin), str(p_bin), str(y_bin), CORE_NUM,
    ]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(BUILD))

    print("=== [3/3] Verify ===")
    subprocess.check_call(
        [sys.executable, str(SCRIPTS / "verify_result.py"), "0"], cwd=str(BUILD)
    )
    print("=== PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        die(f"command failed with exit {e.returncode}: {e.cmd}")
