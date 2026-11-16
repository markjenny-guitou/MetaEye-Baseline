"""Download pinned upstream source, ImageNet weights, or the RegDB archive."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "ADCA": {
        "repository": "yangbincv/ADCA",
        "revision": "8d0a8cb3e74ec3383ebe9f528ae229d7c6bbdab9",
        "files": ["LICENSE", "README.md", "ChannelAug.py", "clustercontrast/models/cm.py",
                  "clustercontrast/utils/faiss_rerank.py", "adca_regdb_train.py", "test_regdb.py"],
    },
    "TransReID": {
        "repository": "damo-cv/TransReID",
        "revision": "dec55046fcdfadee14e2c28e2df89305d8f7557a",
        "files": ["LICENSE", "README.md", "model/backbones/vit_pytorch.py"],
    },
    "SDCL": {
        "repository": "yangbincv/SDCL",
        "revision": "6033332a214112ff6fe4ee1e462f5f14f5106672",
        "files": ["LICENSE", "README.md", "cluster_contrast_vit_regdb.py"],
    },
}
WEIGHT_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
)
REGDB_ID = "1gnVt9GIQSvium_mcxc7AWLhSXm6lNWsa"
REGDB_URL = (
    f"https://drive.usercontent.google.com/download?id={REGDB_ID}&export=download&confirm=t"
)


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    partial = destination.with_name(destination.name + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "MetaEye-Baseline/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError(f"Expected a file, received an HTML page from {url}")
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an unexpected resume offset")
                expected = response.headers.get("Content-Length")
                expected = int(expected) + (offset if resume else 0) if expected else None
                with partial.open("ab" if resume else "wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                if expected and partial.stat().st_size != expected:
                    raise RuntimeError("Incomplete download")
            partial.replace(destination)
            print(f"Saved {destination.relative_to(ROOT)}", flush=True)
            return destination
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def fetch_sources():
    tasks = []
    for name, source in SOURCES.items():
        for filename in source["files"]:
            url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{filename}"
            tasks.append((url, ROOT / "third_party" / name / filename))
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda task: download(*task), tasks))
    # This private PyTorch module was removed in newer releases.
    source_path = ROOT / "third_party/TransReID/model/backbones/vit_pytorch.py"
    source = source_path.read_text(encoding="utf-8")
    source_path.write_text(source.replace("from torch._six import container_abcs", "from collections import abc as container_abcs"), encoding="utf-8")
    manifest = {"sources": SOURCES, "local_patch": "torch._six.container_abcs -> collections.abc"}
    (ROOT / "third_party/sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def fetch_weights():
    target = download(WEIGHT_URL, ROOT / "weights/jx_vit_base_p16_224-80ecf9dd.pth")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith("80ecf9dd"):
        raise RuntimeError(f"ImageNet checkpoint checksum mismatch: {digest}")


def fetch_regdb():
    archive = download(REGDB_URL, ROOT / "data/downloads/RegDB.zip")
    destination = ROOT / "data"
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip():
            raise RuntimeError("RegDB archive checksum failed")
        for info in handle.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe archive member")
        handle.extractall(destination)
    manifest = {
        "archive_url": REGDB_URL,
        "reference": "https://github.com/bismex/HiCMD#training-and-testing",
        "dataset_paper": "https://doi.org/10.3390/s17030605",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    (ROOT / "data/regdb_source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=["sources", "weights", "regdb"])
    parser.add_argument("--proxy", help="Optional HTTP proxy for these downloads only")
    arguments = parser.parse_args()
    if arguments.proxy:
        urllib.request.install_opener(urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": arguments.proxy, "https": arguments.proxy})))
    {"sources": fetch_sources, "weights": fetch_weights, "regdb": fetch_regdb}[arguments.asset]()
