from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torch.nn import functional as F


@torch.no_grad()
def jaccard_distance(features, k1, k2):
    """ADCA's k-reciprocal Jaccard encoding, with exact CPU neighbors instead of GPU FAISS.

    Based on third_party/ADCA/clustercontrast/utils/faiss_rerank.py (MIT).
    This backend targets RegDB-sized per-modality sets.
    """
    features = F.normalize(features.float().cpu(), dim=1)
    size = len(features)
    if size < 2:
        raise ValueError("Clustering requires at least two images")
    k1 = min(k1, size)
    k2 = min(k2, k1)
    distances = (2 - 2 * features @ features.T).clamp_min_(0).numpy()
    rank = np.argsort(distances, axis=1, kind="stable")[:, :k1]

    def reciprocal(index, neighbors):
        forward = rank[index, :neighbors + 1]
        backward = rank[forward, :neighbors + 1]
        return forward[np.any(backward == index, axis=1)]

    full = [reciprocal(index, k1) for index in range(size)]
    half = [reciprocal(index, int(np.around(k1 / 2))) for index in range(size)]
    encoding = np.zeros((size, size), dtype=np.float32)
    for index, neighbors in enumerate(full):
        expanded = list(neighbors)
        for candidate in neighbors:
            candidate_neighbors = half[candidate]
            if len(np.intersect1d(candidate_neighbors, neighbors)) > 2 * len(candidate_neighbors) / 3:
                expanded.extend(candidate_neighbors.tolist())
        expanded = np.unique(expanded)
        weights = np.exp(-distances[index, expanded])
        encoding[index, expanded] = weights / weights.sum()
    if k2 > 1:
        encoding = np.stack([encoding[rank[index, :k2]].mean(axis=0) for index in range(size)])
    inverse = [np.flatnonzero(encoding[:, index]) for index in range(size)]
    result = np.zeros((size, size), dtype=np.float32)
    for index in range(size):
        intersection = np.zeros(size, dtype=np.float32)
        for column in np.flatnonzero(encoding[index]):
            members = inverse[column]
            intersection[members] += np.minimum(encoding[index, column], encoding[members, column])
        result[index] = 1 - intersection / np.maximum(2 - intersection, 1e-12)
    result = np.maximum((result + result.T) * 0.5, 0)
    np.fill_diagonal(result, 0)
    return result


def cluster_features(features, eps, config):
    distance = jaccard_distance(features, config["k1"], config["k2"])
    labels = DBSCAN(eps=eps, min_samples=config["min_samples"], metric="precomputed", n_jobs=1).fit_predict(distance)
    valid = sorted(label for label in np.unique(labels) if label >= 0)
    if len(valid) < 2:
        raise RuntimeError(f"DBSCAN produced {len(valid)} clusters, {sum(labels < 0)} outliers. "
                           "Inspect ImageNet initialization and training-only clustering settings.")
    centers = torch.stack([features[labels == label].mean(dim=0) for label in valid])
    return labels, F.normalize(centers.float(), dim=1)


@torch.no_grad()
def aggregate_memories(visible_features, thermal_features, visible_labels, thermal_labels,
                       visible_centers, thermal_centers, threshold, weight):
    """Instance-pair votes and one-to-one memory aggregation from ADCA's RegDB CMA."""
    similarity = visible_features.float() @ thermal_features.float().T
    # Upstream select_merge_data counts every cross-modal pair above the threshold.
    pairs = torch.nonzero(similarity >= threshold, as_tuple=False).tolist()
    votes = Counter((int(visible_labels[visible]), int(thermal_labels[thermal]))
                    for visible, thermal in pairs
                    if visible_labels[visible] >= 0 and thermal_labels[thermal] >= 0)
    used_visible, used_thermal = set(), set()
    matches = []
    for (visible, thermal), count in votes.most_common():
        if visible in used_visible or thermal in used_thermal:
            continue
        thermal_reference = thermal_centers[thermal].clone()
        visible_centers[visible] = weight * visible_centers[visible] + (1 - weight) * thermal_reference
        used_visible.add(visible)
        used_thermal.add(thermal)
        matches.append({"visible_cluster": visible, "thermal_cluster": thermal, "votes": count})
    return visible_centers, thermal_centers, matches
