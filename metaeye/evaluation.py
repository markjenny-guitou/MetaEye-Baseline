import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FeatureImages, read_split
from .model import ReIDEncoder
from .artifacts import new_archive, retain_file, write_json, write_npz


@torch.inference_mode()
def extract_features(model, records, config, device, flip=False):
    loader = DataLoader(FeatureImages(records, config["height"], config["width"]),
                        batch_size=config["feature_batch_size"], shuffle=False,
                        num_workers=config["workers"], pin_memory=device.type == "cuda")
    model.eval()
    chunks = []
    for images in tqdm(loader, desc="Features", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=config["amp"] and device.type == "cuda"):
            embeddings = model(images)
            if flip:
                embeddings = F.normalize(embeddings + model(images.flip(-1)), dim=1)
        chunks.append(embeddings.float().cpu())
    return torch.cat(chunks)


def retrieval_metrics(similarity, query_ids, gallery_ids):
    """RegDB cross-modal CMC/mAP/mINP; all opposite-modality images are eligible."""
    return retrieval_results(similarity, query_ids, gallery_ids)[0]


def retrieval_results(similarity, query_ids, gallery_ids):
    similarity = np.asarray(similarity)
    if not np.isfinite(similarity).all():
        raise ValueError("Non-finite retrieval scores")
    order = np.argsort(-similarity, axis=1, kind="stable")
    matches = np.asarray(query_ids)[:, None] == np.asarray(gallery_ids)[order]
    rank1, average_precision, inverse_penalty, valid_indices = [], [], [], []
    first_positive, last_positive = [], []
    cmc = np.zeros(matches.shape[1], dtype=np.float64)
    for index, row in enumerate(matches):
        positions = np.flatnonzero(row)
        if len(positions) == 0:
            continue
        valid_indices.append(index)
        first_positive.append(int(positions[0]) + 1)
        last_positive.append(int(positions[-1]) + 1)
        cmc += np.maximum.accumulate(row)
        rank1.append(float(row[0]))
        average_precision.append(float(np.mean(np.arange(1, len(positions) + 1) / (positions + 1))))
        inverse_penalty.append(float(len(positions) / (positions[-1] + 1)))
    if not rank1:
        raise ValueError("No query has a positive gallery image")
    metrics = {"rank1": float(np.mean(rank1)), "mAP": float(np.mean(average_precision)),
               "mINP": float(np.mean(inverse_penalty)), "valid_queries": len(rank1)}
    evidence = {"query_ids": np.asarray(query_ids), "gallery_ids": np.asarray(gallery_ids),
                "top20_indices": order[:, :20],
                "top20_similarity": np.take_along_axis(similarity, order[:, :20], axis=1),
                "valid_query_indices": np.asarray(valid_indices), "ap": np.asarray(average_precision),
                "inp": np.asarray(inverse_penalty), "first_positive_rank": np.asarray(first_positive),
                "last_positive_rank": np.asarray(last_positive), "cmc": cmc / len(rank1)}
    return metrics, evidence


def evaluate_checkpoint(checkpoint_path, root_override=None, feature_batch_size=None):
    # This file is a local training checkpoint produced by this project.
    checkpoint_path = Path(checkpoint_path).resolve()
    torch.set_num_threads(4)
    archive = new_archive(checkpoint_path.parent / "evaluations", "evaluation")
    retained_checkpoint = archive / "checkpoint.pt"
    retain_file(checkpoint_path, retained_checkpoint)
    with retained_checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    checkpoint = torch.load(retained_checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    if root_override:
        config["data_root"] = str(Path(root_override).resolve())
    if feature_batch_size:
        config["feature_batch_size"] = feature_batch_size
    from .data import audit_regdb
    audit_regdb(config["data_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReIDEncoder(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    records = {modal: read_split(config["data_root"], "test", modal, config["trial"])
               for modal in ("visible", "thermal")}
    features = {modal: extract_features(model, values, config, device, flip=True)
                for modal, values in records.items()}
    similarity = (features["visible"] @ features["thermal"].T).numpy()
    visible_ids = [record.identity for record in records["visible"]]
    thermal_ids = [record.identity for record in records["thermal"]]
    visible_metrics, visible_evidence = retrieval_results(similarity, visible_ids, thermal_ids)
    thermal_metrics, thermal_evidence = retrieval_results(similarity.T, thermal_ids, visible_ids)
    result = {
        "checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"],
        "retained_checkpoint": str(retained_checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "trial": config["trial"], "seed": config["seed"], "flip_test": True,
        "protocol": "Single RegDB split, fixed last checkpoint. Final study requires 10 splits.",
        "visible_to_thermal": visible_metrics,
        "thermal_to_visible": thermal_metrics,
        "evidence_directory": str(archive),
    }
    write_npz(archive / "visible_to_thermal.npz", **visible_evidence)
    write_npz(archive / "thermal_to_visible.npz", **thermal_evidence)
    write_json(archive / "metrics.json", result)
    write_json(archive / "config.json", config)
    # Preserve files from older baseline versions before publishing the new latest exports.
    for filename in ("metrics.json", "gallery_features.pt"):
        previous = checkpoint_path.parent / filename
        if previous.is_file():
            retain_file(previous, archive / f"previous_{filename}")
    write_json(checkpoint_path.parent / "metrics.json", result)
    torch.save({"features": features, "paths": {key: [str(record.path) for record in values]
                for key, values in records.items()}, "identities": {"visible": visible_ids, "thermal": thermal_ids},
                "checkpoint": str(retained_checkpoint), "checkpoint_sha256": checkpoint_sha256,
                "trial": config["trial"]},
               archive / "gallery_features.pt")
    latest = checkpoint_path.parent / f"gallery_features_{archive.name}.partial"
    retain_file(archive / "gallery_features.pt", latest)
    latest.replace(checkpoint_path.parent / "gallery_features.pt")
    print(json.dumps(result, indent=2), flush=True)
    return result
