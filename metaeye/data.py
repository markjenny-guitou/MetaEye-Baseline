from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

from .model import load_upstream


@dataclass(frozen=True)
class Record:
    path: Path
    identity: int


def read_split(root, split, modality, trial):
    root = Path(root).resolve()
    manifest = root / "idx" / f"{split}_{modality}_{trial}.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}. Run scripts/fetch_assets.py regdb first.")
    records = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        relative, identity = line.rsplit(maxsplit=1)
        image = (root / relative.replace("\\", "/")).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Invalid image path on line {number} of {manifest}: {image}")
        records.append(Record(image, int(identity)))
    if not records or len({record.path for record in records}) != len(records):
        raise ValueError(f"Empty manifest or duplicate paths: {manifest}")
    return records


def audit_regdb(root, all_trials=True, check_images=False):
    summary = []
    for trial in range(1, 11) if all_trials else [1]:
        parts = {(split, modal): read_split(root, split, modal, trial)
                 for split in ("train", "test") for modal in ("visible", "thermal")}
        # RegDB relabels both train and test to 0..205. Folder IDs retain the original identity.
        identities = {key: {record.path.parent.name for record in values} for key, values in parts.items()}
        if identities["train", "visible"] != identities["train", "thermal"]:
            raise ValueError(f"Training modalities have different identities in trial {trial}")
        if identities["test", "visible"] != identities["test", "thermal"]:
            raise ValueError(f"Test modalities have different identities in trial {trial}")
        if identities["train", "visible"] & identities["test", "visible"]:
            raise ValueError(f"Train/test identity leakage in trial {trial}")
        for split in ("train", "test"):
            mappings = []
            for modal in ("visible", "thermal"):
                by_label = defaultdict(set)
                for record in parts[split, modal]:
                    by_label[record.identity].add(record.path.parent.name)
                if any(len(values) != 1 for values in by_label.values()):
                    raise ValueError(f"Inconsistent folder/label mapping in trial {trial}")
                mappings.append(dict(by_label))
            if mappings[0] != mappings[1]:
                raise ValueError(f"Cross-modal label mapping mismatch in trial {trial}")
        for key, records in parts.items():
            counts = defaultdict(int)
            for record in records:
                counts[record.identity] += 1
            if len(records) != 2060 or len(counts) != 206 or set(counts.values()) != {10}:
                raise ValueError(f"Unexpected RegDB counts in trial {trial}, {key}")
        summary.append({"trial": trial, "train_ids": 206, "test_ids": 206,
                        "train_images": 4120, "test_images": 4120})
    if check_images:
        unique = {record.path for records in parts.values() for record in records}
        for image in sorted(unique):
            with Image.open(image) as handle:
                handle.verify()
    return summary


def image_transform(height, width, training=False, channel=False, thermal=False):
    steps = [T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC)]
    if training:
        steps.extend([T.Pad(10), T.RandomCrop((height, width)), T.RandomHorizontalFlip()])
    steps.extend([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    augmentation = load_upstream("upstream_channel", "third_party/ADCA/ChannelAug.py") if training else None
    if training:
        steps.append(augmentation.ChannelRandomErasing(probability=0.5))
    if channel:
        steps.append(augmentation.ChannelExchange(gray=2))
    if thermal:
        steps.append(augmentation.ChannelAdapGray(probability=0.5))
    return T.Compose(steps)


class FeatureImages(Dataset):
    def __init__(self, records, height, width):
        self.records = records
        self.transform = image_transform(height, width)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index].path) as image:
            return self.transform(image.convert("RGB"))


class TrainingImages(Dataset):
    """Returns cluster labels only. Ground-truth identities are never returned."""

    def __init__(self, records, labels, height, width, visible):
        self.records = records
        self.labels = np.asarray(labels, dtype=np.int64)
        self.visible = visible
        self.transform = image_transform(height, width, training=True, thermal=not visible)
        self.channel_transform = image_transform(height, width, training=True, channel=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index].path) as handle:
            image = handle.convert("RGB")
            if self.visible:
                return self.transform(image), self.channel_transform(image), int(self.labels[index])
            return self.transform(image), int(self.labels[index])


class PseudoIdentityBatchSampler(Sampler):
    def __init__(self, labels, batch_size, instances, iterations, seed):
        if instances < 1 or batch_size % instances or batch_size < 2:
            raise ValueError("batch_size must be >= 2 and divisible by instances")
        self.groups = {int(label): np.flatnonzero(np.asarray(labels) == label)
                       for label in np.unique(labels) if label >= 0}
        self.identities_per_batch = batch_size // instances
        if len(self.groups) < self.identities_per_batch:
            raise ValueError(f"Found {len(self.groups)} pseudo identities, need {self.identities_per_batch}. "
                             "Check clustering or reduce batch_size. Ground-truth labels are not a fallback.")
        self.instances = instances
        self.iterations = iterations
        self.seed = seed

    def __len__(self):
        return self.iterations

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        identities = list(self.groups)
        for _ in range(self.iterations):
            batch = []
            for identity in rng.choice(identities, self.identities_per_batch, replace=False):
                members = self.groups[int(identity)]
                batch.extend(rng.choice(members, self.instances, replace=len(members) < self.instances).tolist())
            yield batch
