"""RAHSM equations (1)-(12), with a deep-only SDCL label association.

Implementation choices absent from the paper are documented in docs/RAHSM复现说明.md.
No ground-truth identity labels are accepted by these functions.
"""

import math

import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F


def validate_rahsm(config):
    method = config.get("method", "adca_transreid")
    if method not in ("adca_transreid", "rahsm"):
        raise ValueError(f"Unknown method: {method}")
    if method != "rahsm":
        return
    settings = config["rahsm"]
    for name in ("chp", "khn", "pts"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"rahsm.{name} must be a boolean")
    for name in ("subclusters", "neighbors", "cross_neighbors", "smoothing_neighbors"):
        if not isinstance(settings[name], int) or isinstance(settings[name], bool) or settings[name] < 1:
            raise ValueError(f"rahsm.{name} must be a positive integer")
    if not math.isfinite(settings["lambda0"]) or settings["lambda0"] < 0:
        raise ValueError("rahsm.lambda0 must be finite and nonnegative")
    if not math.isfinite(config["temperature"]) or config["temperature"] <= 0:
        raise ValueError("temperature must be finite and positive")
    if not 0 <= config["memory_momentum"] <= 1:
        raise ValueError("memory_momentum must lie in [0, 1]")


def progressive_lambda(iteration, total_iterations, lambda0, enabled=True):
    """Equation (10). Iterations are counted within the cross-modal phase."""
    if total_iterations < 1 or not 0 <= iteration <= total_iterations:
        raise ValueError("Expected 0 <= iteration <= total_iterations and total_iterations > 0")
    if not math.isfinite(lambda0) or lambda0 < 0:
        raise ValueError("lambda0 must be finite and nonnegative")
    if not enabled:
        return float(lambda0)
    return float(lambda0 * math.log1p(math.expm1(1.0) * iteration / total_iterations))


def iteration_lambda(config, epoch, step):
    if epoch < config["warmup_epochs"]:
        return 0.0
    settings = config["rahsm"]
    iteration = (epoch - config["warmup_epochs"]) * config["iterations"] + step
    total = (config["epochs"] - config["warmup_epochs"]) * config["iterations"]
    return progressive_lambda(iteration, total, settings["lambda0"], settings["pts"])


def ranked_mode(labels):
    """Majority label, breaking ties by the first (nearest) occurrence."""
    counts = (labels[:, :, None] == labels[:, None, :]).sum(dim=2)
    chosen = counts.argmax(dim=1, keepdim=True)
    return labels.gather(1, chosen).squeeze(1)


@torch.no_grad()
def associate_shared_labels(features, labels, cross_neighbors, smoothing_neighbors):
    """SDCL's cross-ranking vote and within-visible smoothing, using deep features.

    Thermal pseudo labels define the shared space. Keep only identities observed
    in both modalities and keep all original DBSCAN outliers excluded.
    """
    visible = F.normalize(features["visible"].float().cpu(), dim=1)
    thermal = F.normalize(features["thermal"].float().cpu(), dim=1)
    original_visible = np.asarray(labels["visible"], dtype=np.int64)
    original_thermal = np.asarray(labels["thermal"], dtype=np.int64)
    if len(visible) != len(original_visible) or len(thermal) != len(original_thermal):
        raise ValueError("Feature and pseudo-label lengths differ")
    if not len(visible) or not len(thermal) or cross_neighbors < 1 or smoothing_neighbors < 1:
        raise ValueError("Association requires nonempty modalities and positive neighbor counts")
    cross_k = min(cross_neighbors, len(thermal))
    within_k = min(smoothing_neighbors, len(visible))
    cross_order = torch.argsort(visible @ thermal.T, dim=1, descending=True, stable=True)[:, :cross_k]
    votes = torch.as_tensor(original_thermal)[cross_order]
    transferred = ranked_mode(votes)
    within_order = torch.argsort(visible @ visible.T, dim=1, descending=True, stable=True)[:, :within_k]
    smoothed = ranked_mode(transferred[within_order]).numpy()
    smoothed[original_visible < 0] = -1
    shared_ids = sorted((set(smoothed.tolist()) & set(original_thermal.tolist())) - {-1})
    if len(shared_ids) < 2:
        raise RuntimeError(f"Cross-modal association yielded {len(shared_ids)} shared identities; "
                           "inspect training-only clustering and association. No label fallback is used.")
    visible_labels = np.full(len(visible), -1, dtype=np.int64)
    thermal_labels = np.full(len(thermal), -1, dtype=np.int64)
    for dense, original in enumerate(shared_ids):
        visible_labels[smoothed == original] = dense
        thermal_labels[original_thermal == original] = dense
    metadata = {"strategy": "sdcl_deep_rank_vote", "cross_neighbors": cross_k,
                "smoothing_neighbors": within_k, "thermal_id_for_shared_id": shared_ids,
                "shared_identities": len(shared_ids),
                "visible_discarded": int(sum(visible_labels < 0)),
                "thermal_discarded": int(sum(thermal_labels < 0))}
    return {"visible": visible_labels, "thermal": thermal_labels}, metadata


def hard_positives(centers, magnitude):
    """Equations (5)-(8), before any normalization, centers shape [2, C, D]."""
    return centers + magnitude * (centers - centers.flip(0))


class RAHSMBank(nn.Module):
    """Modality means, per-identity K-means subclusters and role-aware logits."""

    def __init__(self, centers, subcenters, identities, modalities, slots, counts, settings, temperature, momentum):
        super().__init__()
        self.register_buffer("centers", centers)
        self.register_buffer("subcenters", subcenters)
        self.register_buffer("identities", identities)
        self.register_buffer("modalities", modalities)
        self.register_buffer("slots", slots)
        self.register_buffer("subcluster_counts", counts)
        self.settings = dict(settings)
        self.temperature = temperature
        self.momentum = momentum
        self.class_count = centers.shape[1]
        self.neighbor_count = min(settings["neighbors"], self.class_count - 1)
        self.slots_per_class = int(counts.sum(dim=0).max()) * self.neighbor_count

    @classmethod
    @torch.no_grad()
    def from_features(cls, features, labels, config, seed):
        settings = config["rahsm"]
        arrays = {modal: features[modal].detach().float().cpu() for modal in ("visible", "thermal")}
        label_arrays = {modal: np.asarray(labels[modal], dtype=np.int64) for modal in arrays}
        classes = sorted(set(label_arrays["visible"].tolist()) - {-1})
        thermal_classes = sorted(set(label_arrays["thermal"].tolist()) - {-1})
        if len(classes) < 2 or classes != thermal_classes or classes != list(range(len(classes))):
            raise ValueError("Shared labels must be contiguous and present in both modalities, with >= 2 identities")
        means, subcenters, owners, modalities, slots = [], [], [], [], []
        counts = torch.zeros(2, len(classes), dtype=torch.long)
        offsets = [0] * len(classes)
        for modality, modal in enumerate(("visible", "thermal")):
            modal_means = []
            for identity in classes:
                samples = arrays[modal][label_arrays[modal] == identity]
                # Equations (1)-(2) use arithmetic means. Do not normalize these means.
                modal_means.append(samples.mean(dim=0))
                if settings["khn"]:
                    unique = torch.unique(samples, dim=0)
                    count = min(settings["subclusters"], len(unique))
                    fitted = KMeans(n_clusters=count, n_init=10, random_state=seed).fit(samples.numpy())
                    components = torch.from_numpy(fitted.cluster_centers_).to(samples.dtype)
                else:
                    count = 1
                    components = samples.mean(dim=0, keepdim=True)
                counts[modality, identity] = count
                subcenters.extend(components.unbind(0))
                owners.extend([identity] * count)
                modalities.extend([modality] * count)
                slots.extend(range(offsets[identity], offsets[identity] + count))
                offsets[identity] += count
            means.append(torch.stack(modal_means))
        return cls(torch.stack(means), torch.stack(subcenters), torch.tensor(owners),
                   torch.tensor(modalities), torch.tensor(slots), counts, settings,
                   config["temperature"], config["memory_momentum"])

    @torch.no_grad()
    def negative_candidates(self, magnitude):
        """Equation (9): nearest other-identity modality means direct each subcluster.

        K2 > 1 generates one candidate per neighbor. Default K2=1 is the explicit
        formula given by the paper. Padding never participates in selection.
        """
        candidates = self.centers.new_zeros(self.class_count, self.slots_per_class, self.centers.shape[-1])
        valid = torch.zeros(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        neighbors = torch.empty(len(self.subcenters), self.neighbor_count, dtype=torch.long, device=candidates.device)
        for modality in (0, 1):
            indices = torch.where(self.modalities == modality)[0]
            components = self.subcenters[indices]
            owners = self.identities[indices]
            prototypes = self.centers[modality]
            distance = (components.square().sum(dim=1, keepdim=True)
                        + prototypes.square().sum(dim=1)[None] - 2 * components @ prototypes.T)
            distance.scatter_(1, owners[:, None], float("inf"))
            nearest = torch.argsort(distance, dim=1, stable=True)[:, :self.neighbor_count]
            generated = components[:, None] + magnitude * (prototypes[nearest] - prototypes[owners, None])
            locations = self.slots[indices, None] * self.neighbor_count + torch.arange(self.neighbor_count, device=candidates.device)
            candidates[owners[:, None], locations] = generated
            valid[owners[:, None], locations] = True
            neighbors[indices] = nearest
        return candidates, valid, neighbors

    def loss(self, queries, targets, modality, magnitude, negatives=None):
        if modality not in (0, 1) or queries.ndim != 2 or targets.shape != (len(queries),):
            raise ValueError("Invalid query shape, targets or modality")
        if not len(queries) or bool((targets < 0).any()) or bool((targets >= self.class_count).any()):
            raise ValueError("Queries must have valid shared pseudo labels")
        if self.settings["chp"]:
            positives = hard_positives(self.centers, magnitude)[:, targets].permute(1, 0, 2)
            positive_distance = (queries[:, None] - positives).square().sum(dim=-1)
            selected_modality = positive_distance.argmax(dim=1)
            positive = positives[torch.arange(len(queries), device=queries.device), selected_modality]
        else:
            selected_modality = torch.full_like(targets, modality)
            positive = self.centers[modality, targets]
        if self.settings["khn"]:
            if negatives is None:
                negatives = self.negative_candidates(magnitude)
            candidates, valid, _ = negatives
            flat = candidates.flatten(0, 1)
            distances = (queries.square().sum(dim=1, keepdim=True) + flat.square().sum(dim=1)[None]
                         - 2 * queries @ flat.T).reshape(len(queries), self.class_count, -1)
            distances = distances.masked_fill(~valid[None], float("inf"))
            selected_slot = distances.argmin(dim=2)
            identity = torch.arange(self.class_count, device=queries.device)[None]
            selected = candidates[identity, selected_slot]
            logits = torch.einsum("bd,bcd->bc", queries, selected)
        else:
            selected_slot = torch.full((len(queries), self.class_count), -1, dtype=torch.long, device=queries.device)
            logits = queries @ self.centers[modality].T
        # Equation (11) replaces the target-class negative by the hardest positive.
        logits = logits.scatter(1, targets[:, None], (queries * positive).sum(dim=1, keepdim=True))
        loss = F.cross_entropy(logits / self.temperature, targets)
        details = {"positive_modality": selected_modality.detach(), "negative_slot": selected_slot.detach(),
                   "positive_squared_distance": (queries - positive).square().sum(dim=1).detach()}
        return loss, details

    @torch.no_grad()
    def update(self, visible_features, visible_targets, thermal_features, thermal_targets):
        """Equations (3)-(4): one batch-mean momentum update per identity/modality."""
        for modality, features, targets in ((0, visible_features, visible_targets),
                                             (1, thermal_features, thermal_targets)):
            for identity in targets.unique():
                mean = features.detach()[targets == identity].mean(dim=0).to(self.centers.dtype)
                self.centers[modality, identity].mul_(self.momentum).add_(mean, alpha=1 - self.momentum)

    def archive_arrays(self):
        return {name: value.detach().cpu().numpy() for name, value in self.named_buffers()}
