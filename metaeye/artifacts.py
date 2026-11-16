"""Preserve experiment inputs and records without replacing older attempts."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid
import zipfile

import numpy as np
import torch


def new_archive(parent, prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(parent) / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def retain_file(source, destination):
    """Source files must subsequently be updated through atomic replacement."""
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        with open(source, "rb") as original, open(destination, "xb") as retained:
            shutil.copyfileobj(original, retained)
        shutil.copystat(source, destination)


def capture_run(output, config, records):
    root = Path(__file__).resolve().parents[1]
    archive = new_archive(Path(output) / "provenance", "run")
    files = [root / "main.py", root / "requirements.txt", root / "requirements.lock.txt"]
    files += sorted((root / "metaeye").glob("*.py"))
    files += sorted((root / "configs").glob("*.json"))
    files += sorted((root / "third_party").rglob("*.py"))
    files += sorted((root / "third_party").rglob("LICENSE"))
    files += [root / "third_party/sources.json"]
    source_hashes = {}
    with zipfile.ZipFile(archive / "source.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        for file in files:
            if file.is_file():
                name = file.relative_to(root).as_posix()
                source_hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
                handle.write(file, name)
    for split in ("train", "test"):
        for modal in ("visible", "thermal"):
            filename = f"{split}_{modal}_{config['trial']}.txt"
            shutil.copy2(Path(config["data_root"]) / "idx" / filename, archive / filename)
    samples = {modal: [record.path.relative_to(Path(config["data_root"])).as_posix()
                       for record in values] for modal, values in records.items()}
    write_json(archive / "training_samples.json", samples)
    write_json(archive / "config.json", config)
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow")}
    environment = {"python": sys.version, "platform": platform.platform(), "packages": packages,
                   "cuda_runtime": torch.version.cuda, "source_sha256": source_hashes}
    pretrained = Path(config["pretrained"])
    if pretrained.is_file():
        with pretrained.open("rb") as handle:
            environment["pretrained_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        environment.update(gpu=properties.name, vram_bytes=properties.total_memory)
    write_json(archive / "environment.json", environment)
    return archive
