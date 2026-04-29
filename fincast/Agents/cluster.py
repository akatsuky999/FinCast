from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np

from fincast.tools.similarity import comprehensive_similarity, zscore


def cluster_cases(
    cases: list[dict[str, Any]],
    num_clusters: int = 6,
    method: str = "weighted",
    min_votes: int = 3,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Cluster cases with K-Medoids and return per-cluster model weight distributions.

    Ported from AlphaCast alphacast/tools/analysis.py:cluster_by_kmedoid().

    Args:
        cases: List of case dicts, each with 'zscored_window' and 'best_model' keys.
        num_clusters: Number of K-Medoids clusters.
        method: 'voting' (single best model) or 'weighted' (model weight distribution).
        min_votes: Minimum votes to retain a model in weighted mode.
        seed: Random seed for initial medoid selection.

    Returns:
        List of cluster dicts:
        {
            "cluster_id": int,
            "medoid_window": list[float],
            "model_weights": dict[str, int],
            "total_weight": int,
            "member_count": int,
        }
    """
    if not cases:
        return []

    try:
        from pyclustering.cluster.kmedoids import kmedoids
    except ImportError:
        raise ImportError("pyclustering is required for K-Medoids clustering. Install with: pip install pyclustering")

    k = int(num_clusters) if num_clusters and num_clusters > 0 else 4
    k = max(1, min(k, len(cases)))

    window_vectors = [np.asarray(c.get("zscored_window", []), dtype=float).tolist() for c in cases]

    random.seed(seed)
    initial_medoids = random.sample(range(len(cases)), k)

    kmedoids_instance = kmedoids(window_vectors, initial_medoids, ccore=False)
    kmedoids_instance.process()

    cluster_indices = kmedoids_instance.get_clusters()
    medoid_indices = kmedoids_instance.get_medoids()

    clusters: list[dict[str, Any]] = []

    for gi, medoid_idx in enumerate(medoid_indices):
        medoid_window = np.asarray(cases[medoid_idx].get("zscored_window", []), dtype=float).tolist()
        member_indices = cluster_indices[gi]
        group_cases = [cases[idx] for idx in member_indices]

        if not group_cases:
            clusters.append({
                "cluster_id": gi,
                "medoid_window": medoid_window,
                "model_weights": {},
                "total_weight": 0,
                "member_count": 0,
            })
            continue

        counts = Counter(c.get("best_model", "unknown") for c in group_cases)

        if method == "voting":
            best = counts.most_common(1)[0][0]
            clusters.append({
                "cluster_id": gi,
                "medoid_window": medoid_window,
                "model_weights": {best: 1},
                "total_weight": 1,
                "member_count": len(group_cases),
            })
        elif method == "weighted":
            filtered = {model: count for model, count in counts.items() if count > min_votes}
            clusters.append({
                "cluster_id": gi,
                "medoid_window": medoid_window,
                "model_weights": filtered,
                "total_weight": sum(filtered.values()),
                "member_count": len(group_cases),
            })

    return clusters


def match_cluster(
    query: np.ndarray,
    clusters: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Match a query window to the best cluster via comprehensive similarity.

    Args:
        query: Raw window values (1-D numpy array).
        clusters: List of cluster dicts from cluster_cases().

    Returns:
        Best-matching cluster dict with added 'match_similarity' field, or None.
    """
    if not clusters:
        return None

    query_z = zscore(query)
    best_cluster = None
    best_sim = -1.0

    for cluster in clusters:
        medoid = np.asarray(cluster.get("medoid_window", []), dtype=float)
        if medoid.size == 0:
            continue
        sim = comprehensive_similarity(query_z, medoid)
        if sim > best_sim:
            best_sim = sim
            best_cluster = dict(cluster)

    if best_cluster is not None:
        best_cluster["match_similarity"] = float(best_sim)
    return best_cluster


__all__ = ["cluster_cases", "match_cluster"]
