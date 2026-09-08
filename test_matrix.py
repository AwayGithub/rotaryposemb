import numpy as np
import os
import subprocess
import sys

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'rotary_pos_emb_custom')
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mtest')
os.makedirs(TMP, exist_ok=True)

f32 = np.float32
f16 = np.float16


def golden(x, p, theta):
    orig = x.dtype
    dim = x.shape[-1]
    xf = x.astype(np.float32)
    pf = p.astype(np.float32)
    invf = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[:dim // 2] / dim))
    freqs = pf[..., None] * invf
    tf_ = np.concatenate([freqs, freqs], axis=-1)
    c = np.cos(tf_)[..., None, :]
    s = np.sin(tf_)[..., None, :]
    h = dim // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    y = xf * c + rot.astype(np.float32) * s
    if orig == np.float16:
        return y.astype(np.float16)
    return y.astype(np.float32)


def run_case(dtype, shape, theta, pgen, rng, tag):
    B, S, H, D = shape
    dt = f32 if dtype == 0 else f16
    x = (rng.standard_normal(shape)).astype(dt)
    # mix scales
    x = x * (10.0 ** rng.uniform(-1, 1))
    p = pgen(rng, (B, S))
    xp = os.path.join(TMP, f'{tag}_x.bin')
    pp = os.path.join(TMP, f'{tag}_p.bin')
    yp = os.path.join(TMP, f'{tag}_y.bin')
    x.astype(np.float32 if dtype == 0 else np.float16).tofile(xp)
    p.astype(np.int32).tofile(pp)
    r = subprocess.run([BIN, str(dtype), str(B), str(S), str(H), str(D), str(theta), xp, pp, yp, '0'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'[{tag}] RUN FAIL: {r.stderr.strip()[:200]}')
        return False
    y = np.fromfile(yp, dtype=dt).reshape(shape)
    g = golden(x, p, theta)
    tol = 1e-3 if dtype == 1 else 1e-4
    diff = np.abs(y.astype(np.float32) - g.astype(np.float32))
    tol_arr = tol + tol * np.abs(g.astype(np.float32))
    ok = np.isclose(y.astype(np.float32), g.astype(np.float32), rtol=tol, atol=tol, equal_nan=True)
    nbad = (~ok).sum()
    print(f'[{tag}] shape={shape} dt={dt.__name__} theta={theta} maxdiff={diff.max():.3e} '
          f'bad={nbad}/{y.size} {"" if nbad == 0 else "FAIL"}')
    return nbad == 0


def main():
    rng = np.random.default_rng(12345)
    allok = True
    cases = []

    # small debug
    cases.append((0, (1, 7, 1, 16), 10000.0, lambda rng, sh: rng.integers(0, 50, sh)))
    cases.append((0, (1, 8, 1, 16), 10000.0, lambda rng, sh: rng.integers(0, 50, sh)))
    # fp16 small
    cases.append((1, (1, 8, 1, 16), 10000.0, lambda rng, sh: rng.integers(0, 50, sh)))
    # 非对齐 dim/H + odd half cases
    cases.append((1, (2, 16, 3, 24), 10000.0, lambda rng, sh: rng.integers(0, 8192, sh)))
    cases.append((1, (2, 7, 5, 18), 10000.0, lambda rng, sh: rng.integers(0, 8192, sh)))
    cases.append((0, (3, 11, 2, 20), 10000.0, lambda rng, sh: rng.integers(0, 8192, sh)))
    # p=0 rows and nan/inf
    def pzero(rng, sh):
        p = rng.integers(0, 100, sh)
        p[..., 0] = 0
        return p
    cases.append((1, (1, 9, 2, 16), 10000.0, pzero))
    cases.append((0, (1, 9, 2, 16), 10000.0, pzero))
    # wide positions up to 131072
    cases.append((1, (2, 512, 128, 64), 10000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    # huge positions fp32
    cases.append((0, (2, 64, 8, 64), 10000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    # deepseek-like fp16
    cases.append((1, (2, 512, 128, 64), 10000.0, lambda rng, sh: rng.integers(0, 8192, sh)))
    # theta variations
    cases.append((0, (1, 64, 4, 32), 1.0, lambda rng, sh: rng.integers(0, 1000, sh)))
    cases.append((1, (2, 32, 8, 64), 1000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    cases.append((1, (2, 32, 8, 64), 500000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    cases.append((1, (2, 32, 8, 64), 3.5e4, lambda rng, sh: rng.integers(0, 131072, sh)))
    cases.append((0, (1, 128, 4, 256), 10000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    cases.append((0, (1, 16, 1, 1024), 10000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    cases.append((1, (4, 1024, 32, 128), 10000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    # H=1 big seq
    cases.append((1, (1, 4096, 1, 64), 10000.0, lambda rng, sh: rng.integers(0, 131072, sh)))
    # single row
    cases.append((0, (1, 1, 1, 16), 10000.0, lambda rng, sh: rng.integers(0, 50, sh)))

    for i, (dt, shape, theta, pgen) in enumerate(cases):
        ok = run_case(dt, shape, theta, pgen, rng, f'c{i}')
        allok &= ok
    print('ALL PASS' if allok else 'SOME FAILED')
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
