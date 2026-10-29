"""
Shared feature code. Used by train.py AND by your live app (the saved model imports it).
Keep this file next to the app that calls joblib.load(...).

Input : array (n, 42) = 21 hand landmarks x (x, y), wrist first (any scale / any origin / either hand)
Output: array (n, F) of orientation-aware, scale-invariant features
"""
import numpy as np

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
_IU = np.triu_indices(21, 1)


def canonicalize(X):
    """Wrist -> origin, left hand mirrored to right, scaled by wrist->middle-knuckle length."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    P = X.reshape(-1, 21, 2).copy()
    P -= P[:, :1, :]
    cross = P[:, 5, 0] * P[:, 17, 1] - P[:, 5, 1] * P[:, 17, 0]
    P[cross < 0, :, 0] *= -1
    size = np.linalg.norm(P[:, 9, :], axis=1)
    size = np.where(size < 1e-6, 1.0, size)
    return P / size[:, None, None]


def _angle(P, a, b, c):
    v1, v2 = P[:, a] - P[:, b], P[:, c] - P[:, b]
    return (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)


def landmark_features(X):
    P = canonicalize(X)
    n = len(P)
    coords = P[:, 1:, :].reshape(n, -1)                                   # 40
    d = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
    dists = d[:, _IU[0], _IU[1]]                                          # 210 pairwise distances
    angles = np.stack([_angle(P, a, b, c) for f in FINGERS
                       for a, b, c in ((f[0], f[1], f[2]), (f[1], f[2], f[3]))], axis=1)  # 10 bend angles
    return np.hstack([coords, dists, angles])