from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return np.zeros_like(x)
    return (x - mu) / (sigma + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    return float(np.linalg.norm(a - b))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    dtw_matrix = np.full((n + 1, m + 1), float("inf"))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return dtw_matrix[n, m] / max(n, m)


def trend_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0

    diff_a = np.diff(a)
    diff_b = np.diff(b)

    signs_a = np.sign(diff_a)
    signs_b = np.sign(diff_b)

    direction_consistency = np.mean(signs_a == signs_b)

    strength_a = np.std(diff_a) if len(diff_a) > 0 else 0
    strength_b = np.std(diff_b) if len(diff_b) > 0 else 0

    if strength_a + strength_b == 0:
        strength_similarity = 1.0
    else:
        strength_similarity = 1.0 - abs(strength_a - strength_b) / (strength_a + strength_b)

    return 0.7 * direction_consistency + 0.3 * strength_similarity


def volatility_similarity(a: np.ndarray, b: np.ndarray) -> float:
    vol_a = np.std(a) if len(a) > 0 else 0
    vol_b = np.std(b) if len(b) > 0 else 0

    if vol_a + vol_b == 0:
        return 1.0

    return 1.0 - abs(vol_a - vol_b) / (vol_a + vol_b)


def pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0

    def autocorr(x, max_lag=None):
        if max_lag is None:
            max_lag = min(len(x) // 4, 10)

        x_centered = x - np.mean(x)
        autocorrs = []

        for lag in range(1, max_lag + 1):
            if lag >= len(x):
                break
            corr = np.corrcoef(x_centered[:-lag], x_centered[lag:])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0.0)

        return np.array(autocorrs)

    autocorr_a = autocorr(a)
    autocorr_b = autocorr(b)

    if len(autocorr_a) == 0 or len(autocorr_b) == 0:
        return 0.0

    min_len = min(len(autocorr_a), len(autocorr_b))
    autocorr_a = autocorr_a[:min_len]
    autocorr_b = autocorr_b[:min_len]

    return cosine_similarity(autocorr_a, autocorr_b)


def comprehensive_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    query_norm = zscore(query)
    candidate_norm = zscore(candidate)

    shape_sim = cosine_similarity(query_norm, candidate_norm)

    dtw_dist = dtw_distance(query_norm, candidate_norm)
    dtw_sim = 1.0 / (1.0 + dtw_dist)

    trend_sim = trend_similarity(query, candidate)
    vol_sim = volatility_similarity(query, candidate)
    pattern_sim = pattern_similarity(query, candidate)

    weights = {
        "shape": 0.25,
        "dtw": 0.25,
        "trend": 0.20,
        "volatility": 0.15,
        "pattern": 0.15,
    }

    total_sim = (
        weights["shape"] * max(0, shape_sim)
        + weights["dtw"] * dtw_sim
        + weights["trend"] * trend_sim
        + weights["volatility"] * vol_sim
        + weights["pattern"] * max(0, pattern_sim)
    )

    return total_sim


def match_best_cluster(
    query: np.ndarray, clusters: list[dict], top_k: int = 1
) -> list[dict]:
    """Match query window to the most similar cluster(s) using comprehensive similarity.

    Args:
        query: Raw window values (1-D numpy array).
        clusters: List of cluster dicts, each with 'medoid_window' key.
        top_k: Number of best clusters to return.

    Returns:
        List of cluster dicts with added 'match_similarity' field, sorted descending.
    """
    if not clusters:
        return []
    scored = []
    for cluster in clusters:
        medoid = np.asarray(cluster.get("medoid_window", []), dtype=float)
        if medoid.size == 0:
            continue
        sim = comprehensive_similarity(query, medoid)
        entry = dict(cluster)
        entry["match_similarity"] = float(sim)
        scored.append(entry)
    scored.sort(key=lambda c: c["match_similarity"], reverse=True)
    return scored[:top_k]


def retrieve_similar_cases(
    query: np.ndarray,
    cases: list[dict],
    top_k: int = 20,
    window_key: str = "zscored_window",
) -> list[dict]:
    """Retrieve top-k similar cases by comprehensive similarity.

    Args:
        query: Raw window values (1-D numpy array).
        cases: List of case dicts, each with a window array key.
        top_k: Number of cases to return.
        window_key: Key in each case dict holding the zscored window array.

    Returns:
        List of case dicts with added 'similarity_score' and 'similarity_weight' fields.
    """
    if not cases:
        return []
    query_z = zscore(query)
    scored = []
    for case in cases:
        window = np.asarray(case.get(window_key, []), dtype=float)
        if window.size == 0:
            continue
        sim = comprehensive_similarity(query_z, window)
        case_copy = dict(case)
        case_copy["similarity_score"] = float(sim)
        case_copy["similarity_weight"] = float(1.0 / (1.0 + (1.0 - sim)))
        scored.append(case_copy)
    scored.sort(key=lambda c: c["similarity_score"], reverse=True)
    return scored[:top_k]


__all__ = [
    "comprehensive_similarity",
    "cosine_similarity",
    "dtw_distance",
    "euclidean_distance",
    "match_best_cluster",
    "pattern_similarity",
    "retrieve_similar_cases",
    "trend_similarity",
    "volatility_similarity",
    "zscore",
]
