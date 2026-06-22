"""
aggregators.py — Point-aggregation / localization-recovery methods
==================================================================
All methods take a set of 2D location points (k x 2, in a *metric* CRS) and
return a single estimated location plus optional sparse-recovery diagnostics.

Methods implemented (proposal report sections 3-5, 4B):
  Baselines : mean, median, geometric_median, kmeans, dbscan
  Paper     : nsal                         (Gaussian affinity + MinCut + degree center)
  CS convex : l1_sor                       (group-sparse outlier recovery, section 4.5-4.8)
  CS greedy : omp / cosamp / sp            (annihilator reduction z = F y, section 4B)
  Proposed  : uspa                         (NSAL degree weight x reliability, section 5)

k-guard (section 4.10): the sparse methods only attempt recovery when there are
enough observations to outvote the outliers (k >= K_MIN). Below that they fall
back to the geometric median and flag ``fallback_triggered``.

Every aggregator returns an :class:`AggResult` so per-sign logging is uniform.
All math is translation-equivariant; the numerically sensitive solvers center
the coordinates internally for stability and add the offset back at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import null_space
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans

# Minimum observations before sparse recovery is trusted (section 4.10).
# From the error-correction bound N >= 2s + 1 with s = 3  ->  K_MIN = 7.
K_MIN = 7


@dataclass
class AggResult:
    """Uniform return type for every aggregator (drives per-sign logging)."""

    center: np.ndarray                       # estimated location in the input CRS (2,)
    outlier_idx: list = field(default_factory=list)
    outlier_scores: Optional[np.ndarray] = None   # per-observation ||e_i||_2
    residual_norm: Optional[float] = None
    n_iter: Optional[int] = None
    assumed_s: Optional[int] = None
    fallback_triggered: bool = False


# ---------------------------------------------------------------------------
# Simple / robust baselines
# ---------------------------------------------------------------------------

def mean(coords: np.ndarray) -> AggResult:
    return AggResult(center=coords.mean(axis=0))


def median(coords: np.ndarray) -> AggResult:
    """Coordinate-wise median."""
    return AggResult(center=np.median(coords, axis=0))


def geometric_median(coords: np.ndarray, eps: float = 1e-8,
                     max_iter: int = 200) -> AggResult:
    """Weiszfeld's algorithm for the L2 geometric median (robust 2D center)."""
    x = coords.mean(axis=0)
    for it in range(max_iter):
        d = np.linalg.norm(coords - x, axis=1)
        near = d < eps
        if near.any():
            # x sits on a data point; that point is the median.
            return AggResult(center=coords[near][0], n_iter=it)
        w = 1.0 / d
        x_new = (coords * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < eps:
            x = x_new
            break
        x = x_new
    return AggResult(center=x, n_iter=it)


def kmeans(coords: np.ndarray, n_clusters: int = 2, seed: int = 0) -> AggResult:
    """Cluster baseline: KMeans, keep the largest cluster, return its centroid."""
    k = len(coords)
    if k <= n_clusters:
        return AggResult(center=coords.mean(axis=0))
    labels = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=seed).fit_predict(coords)
    biggest = np.argmax(np.bincount(labels))
    sel = coords[labels == biggest]
    return AggResult(center=sel.mean(axis=0),
                     outlier_idx=np.where(labels != biggest)[0].tolist())


def dbscan(coords: np.ndarray, eps: float = 5.0, min_samples: int = 2) -> AggResult:
    """Density baseline: largest DBSCAN cluster centroid (noise = -1 dropped)."""
    k = len(coords)
    if k <= 2:
        return AggResult(center=coords.mean(axis=0))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    valid = set(labels) - {-1}
    if not valid:
        return AggResult(center=coords.mean(axis=0))
    best = max(valid, key=lambda l: np.sum(labels == l))
    sel = coords[labels == best]
    return AggResult(center=sel.mean(axis=0),
                     outlier_idx=np.where(labels != best)[0].tolist())


# ---------------------------------------------------------------------------
# Paper method: NSAL
# ---------------------------------------------------------------------------

def _gaussian_affinity(coords: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    sq = cdist(coords, coords, metric="sqeuclidean")
    return np.exp(-sq / (2 * sigma ** 2))


def nsal(coords: np.ndarray, sigma: float = 2.5) -> AggResult:
    """Noise & Sparsity Adaptive Localization (paper, section 3).

    Gaussian affinity -> MinCut-style removal of low-degree points ->
    degree-weighted center. Mirrors the validated ``nsal_cluster`` in
    ``eval_table2_localization.py``.
    """
    k = len(coords)
    if k == 1:
        return AggResult(center=coords[0])

    W = _gaussian_affinity(coords, sigma)
    weight_sum = W.sum(axis=1)

    # MinCut surrogate: drop points whose degree jump is anomalously large.
    sorted_w = np.sort(weight_sum)
    diffs = np.diff(sorted_w)
    mean_diff = diffs.mean() if len(diffs) else 0.0
    count = 0
    for diff in diffs:
        if diff > 3 * mean_diff:
            count += 1
        else:
            break
    num_del = min(int(k * 0.3), count)

    ws = weight_sum.copy()
    is_outlier = np.zeros(k, dtype=bool)
    for _ in range(num_del):
        idx = int(np.argmin(ws))
        is_outlier[idx] = True
        ws[idx] = np.inf

    keep = coords[~is_outlier]
    if len(keep) == 0:
        center = coords.mean(axis=0)
    else:
        w = weight_sum[~is_outlier]
        tot = w.sum()
        if tot == 0:
            center = keep.mean(axis=0)
        else:
            norm_w = w / tot * len(keep)
            center = (keep * norm_w[:, None]).mean(axis=0)
    return AggResult(center=center, outlier_idx=np.where(is_outlier)[0].tolist())


# ---------------------------------------------------------------------------
# CS convex: L1-SOR (group-sparse outlier recovery)
# ---------------------------------------------------------------------------

def l1_sor(coords: np.ndarray, lam: float = 2.0, n_iter: int = 100,
           tol: float = 1e-6) -> AggResult:
    """L1 Sparse Outlier Recovery, group-sparse form (section 4.6-4.8).

        min_{l, e_i}  1/2 sum_i ||p_i - l - e_i||^2 + lam sum_i ||e_i||_2

    Solved by alternating minimization: group soft-thresholding of the
    residuals (prox of the group-L1 penalty) followed by a mean update of l.
    """
    p = np.asarray(coords, float)
    offset = p.mean(axis=0)
    p = p - offset                                    # center for stability

    res = geometric_median(p)
    l = res.center
    e = np.zeros_like(p)
    it = 0
    for it in range(1, n_iter + 1):
        r = p - l                                     # residuals
        rn = np.linalg.norm(r, axis=1)
        shrink = np.where(rn > 1e-12,
                          np.clip(1.0 - lam / np.maximum(rn, 1e-12), 0.0, None),
                          0.0)
        e = shrink[:, None] * r                       # group soft-threshold
        l_new = (p - e).mean(axis=0)
        if np.linalg.norm(l_new - l) < tol:
            l = l_new
            break
        l = l_new

    scores = np.linalg.norm(e, axis=1)
    outliers = np.where(scores > 1e-9)[0].tolist()
    resid = float(np.linalg.norm(p - l - e))
    return AggResult(center=l + offset, outlier_idx=outliers,
                     outlier_scores=scores, residual_norm=resid, n_iter=it)


# ---------------------------------------------------------------------------
# CS greedy: OMP / CoSaMP / SP via annihilator reduction  z = F y
# ---------------------------------------------------------------------------

def _build_sensing(k: int, sensing: str, seed: int) -> np.ndarray:
    """Block sensing operator on the stacked vector y in R^{2k}.

    A in R^{2k x 2} repeats the unknown location, so the location lives in
    range(A). F annihilates it: z = F y = F e + F eps. ``annihilator`` uses an
    orthonormal basis of null(A^T); ``gaussian`` randomizes it for incoherence
    via Phi = R P_{A_perp} (section 4B.2).
    """
    A = np.tile(np.eye(2), (k, 1))                    # (2k, 2)
    F = null_space(A.T).T                             # (2k-2, 2k), F A = 0
    if sensing == "annihilator":
        return F
    # Random Gaussian projection living in the annihilator subspace.
    rng = np.random.default_rng(seed)
    P_perp = F.T @ F                                  # projector onto null(A^T)
    R = rng.standard_normal((2 * k - 2, 2 * k))
    return R @ P_perp


def _blocks(k: int):
    """Column index pairs (x_i, y_i) for each observation in y."""
    return [(2 * i, 2 * i + 1) for i in range(k)]


def _greedy_support(coords, s, method, sensing, seed):
    """Recover the outlier observation support via a block greedy pursuit.

    Returns (support set of observation indices, e_hat (k,2), residual_norm).
    Outliers are recovered as whole 2D observations (block sparsity), matching
    the assumption that a frame is an outlier in both coordinates at once.
    """
    k = len(coords)
    offset = coords.mean(axis=0)
    y = (coords - offset).reshape(-1)                 # (2k,)
    Phi = _build_sensing(k, sensing, seed)            # (m, 2k)
    z = Phi @ y
    blocks = _blocks(k)

    def block_corr(residual):
        # ||Phi_block^T r|| for every observation block.
        return np.array([np.linalg.norm(Phi[:, list(b)].T @ residual)
                         for b in blocks])

    def lstsq_on(support):
        cols = [c for i in support for c in blocks[i]]
        if not cols:
            return np.zeros(0), z.copy()
        sub = Phi[:, cols]
        sol, *_ = np.linalg.lstsq(sub, z, rcond=None)
        r = z - sub @ sol
        return sol, r

    support: list = []
    r = z.copy()
    if method == "omp":
        for _ in range(s):
            corr = block_corr(r)
            corr[support] = -1
            j = int(np.argmax(corr))
            support.append(j)
            _, r = lstsq_on(support)
    elif method in ("cosamp", "sp"):
        add = 2 * s if method == "cosamp" else s
        for _ in range(max_greedy_iters(s)):
            proxy = block_corr(r)
            merged = sorted(set(support) | set(np.argsort(proxy)[::-1][:add].tolist()))
            sol, _ = lstsq_on(merged)
            # block energies on the merged support, prune to the strongest s
            energy = {}
            for pos, i in enumerate(merged):
                energy[i] = np.linalg.norm(sol[2 * pos:2 * pos + 2])
            new_support = sorted(sorted(energy, key=energy.get)[::-1][:s])
            _, r = lstsq_on(new_support)
            if new_support == support:
                support = new_support
                break
            support = new_support
    else:
        raise ValueError(f"unknown greedy method: {method}")

    # Reconstruct e_hat on the recovered support for diagnostics.
    sol, r = lstsq_on(support)
    e_hat = np.zeros((k, 2))
    for pos, i in enumerate(support):
        e_hat[i] = sol[2 * pos:2 * pos + 2]
    return set(support), e_hat, float(np.linalg.norm(r))


def max_greedy_iters(s: int) -> int:
    return max(3, 2 * s)


def _greedy(coords, s, method, sensing="gaussian", seed=0) -> AggResult:
    k = len(coords)
    support, e_hat, resid = _greedy_support(coords, s, method, sensing, seed)
    inliers = [i for i in range(k) if i not in support]
    if not inliers:                                   # degenerate: keep everything
        inliers = list(range(k))
        support = set()
    center = coords[inliers].mean(axis=0)
    scores = np.linalg.norm(e_hat, axis=1)
    return AggResult(center=center, outlier_idx=sorted(support),
                     outlier_scores=scores, residual_norm=resid,
                     assumed_s=s)


def omp(coords, s=2, sensing="gaussian", seed=0):
    return _greedy(coords, s, "omp", sensing, seed)


def cosamp(coords, s=2, sensing="gaussian", seed=0):
    return _greedy(coords, s, "cosamp", sensing, seed)


def sp(coords, s=2, sensing="gaussian", seed=0):
    return _greedy(coords, s, "sp", sensing, seed)


# ---------------------------------------------------------------------------
# Proposed: USPA (Uncertainty-aware Sparse Point Aggregation)
# ---------------------------------------------------------------------------

def reliability_score(meta: dict, k: int, variant: str = "area_depth",
                      lambda_d: float = 0.02) -> np.ndarray:
    """Per-observation reliability q_i (section 5.5). Falls back to 1.0 when a
    cue is missing so the method degrades gracefully to plain NSAL."""
    depth = np.asarray(meta.get("depth", np.zeros(k)), float)
    area = np.asarray(meta.get("area", np.ones(k)), float)
    conf = np.asarray(meta.get("conf", np.ones(k)), float)
    area_norm = area / (area.max() + 1e-9)

    if variant == "none":
        return np.ones(k)
    if variant == "depth":
        return np.exp(-lambda_d * depth)
    if variant == "area":
        return area_norm
    if variant == "area_depth":
        return area_norm * np.exp(-lambda_d * depth)
    if variant == "conf_area_depth":
        return conf * area_norm * np.exp(-lambda_d * depth)
    raise ValueError(f"unknown reliability variant: {variant}")


def uspa(coords: np.ndarray, meta: Optional[dict] = None,
         variant: str = "area_depth", sigma: float = 2.5,
         lambda_d: float = 0.02) -> AggResult:
    """NSAL graph degree x observation reliability (section 5.6-5.8)."""
    k = len(coords)
    if k == 1:
        return AggResult(center=coords[0])
    meta = meta or {}
    W = _gaussian_affinity(coords, sigma)
    d_graph = W.sum(axis=1)                            # spatial support
    q = reliability_score(meta, k, variant, lambda_d)  # reliability
    w = d_graph * q
    tot = w.sum()
    if tot <= 0:
        center = coords.mean(axis=0)
    else:
        center = (coords * (w / tot)[:, None]).sum(axis=0)
    return AggResult(center=center)


# ---------------------------------------------------------------------------
# k-guard dispatch (section 4.10)
# ---------------------------------------------------------------------------

SPARSE_METHODS = {"l1_sor", "omp", "cosamp", "sp"}


def aggregate(method: str, coords: np.ndarray, meta: Optional[dict] = None,
              *, s_user: int = 3, K_min: int = K_MIN, **kwargs) -> AggResult:
    """Single entry point with the k-guard applied to the sparse methods.

    For l1_sor/omp/cosamp/sp: if k < K_min the problem is under-determined for
    outlier separation, so fall back to the geometric median and flag it. The
    assumed sparsity is capped at floor((k-1)/2) (error-correction limit).
    """
    coords = np.asarray(coords, float)
    k = len(coords)

    if method in SPARSE_METHODS and k < K_min:
        res = geometric_median(coords)
        res.fallback_triggered = True
        return res

    if method == "mean":
        return mean(coords)
    if method == "median":
        return median(coords)
    if method == "geometric_median":
        return geometric_median(coords)
    if method == "kmeans":
        return kmeans(coords, **_filter(kwargs, ("n_clusters", "seed")))
    if method == "dbscan":
        return dbscan(coords, **_filter(kwargs, ("eps", "min_samples")))
    if method == "nsal":
        return nsal(coords, **_filter(kwargs, ("sigma",)))
    if method == "uspa":
        return uspa(coords, meta, **_filter(kwargs, ("variant", "sigma", "lambda_d")))
    if method == "l1_sor":
        return l1_sor(coords, **_filter(kwargs, ("lam", "n_iter", "tol")))
    if method in ("omp", "cosamp", "sp"):
        s = min(s_user, (k - 1) // 2)                 # cap at correction limit
        s = max(1, s)
        fn = {"omp": omp, "cosamp": cosamp, "sp": sp}[method]
        return fn(coords, s=s, **_filter(kwargs, ("sensing", "seed")))
    raise ValueError(f"unknown method: {method}")


def _filter(kwargs: dict, allowed) -> dict:
    return {k: v for k, v in kwargs.items() if k in allowed}


ALL_METHODS = [
    "mean", "median", "geometric_median", "kmeans", "dbscan",
    "nsal", "l1_sor", "omp", "cosamp", "sp", "uspa",
]
