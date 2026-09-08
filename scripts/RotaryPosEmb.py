#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
RotaryPosEmb golden — matches official PyTorch rotate-half definition.

inv_freq / cos / sin computed in FP32, then cast back to input dtype.
"""
import numpy as np


def impl(x, positions, theta=10000.0):
    """RotaryPosEmb golden.

    x: (B, S, H, rope_dim) float16|float32
    positions: (B, S) int32
    theta: float
    returns y same shape/dtype as x
    """
    orig_dtype = x.dtype
    dim = x.shape[-1]

    x_f = x.astype(np.float32)
    pos_f = positions.astype(np.float32)

    inv_freq = 1.0 / (
        theta ** (np.arange(0, dim, 2, dtype=np.float32)[: dim // 2] / dim)
    )

    freqs = pos_f[..., None] * inv_freq
    theta_freqs = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(theta_freqs)[..., None, :]
    sin = np.sin(theta_freqs)[..., None, :]

    def rotate_half(t):
        t1, t2 = t[..., : dim // 2], t[..., dim // 2 :]
        return np.concatenate([-t2, t1], axis=-1)

    y = x_f * cos + rotate_half(x_f) * sin

    if orig_dtype == np.float16:
        return y.astype(np.float16)
    return y.astype(np.float32)
