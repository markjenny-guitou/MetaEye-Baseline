"""Run doctor, dataset validation, unsupervised training, and RegDB evaluation."""

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from metaeye.data import audit_regdb
from metaeye.evaluation import evaluate_checkpoint
from metaeye.model import ROOT
from metaeye.training import train
from metaeye.rahsm import validate_rahsm


def read_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_root", "output_dir", "pretrained"):
        config[key] = str((ROOT / config[key]).resolve())
    return config


def doctor(config):
    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "pretrained_available": Path(config["pretrained"]).is_file(),
              "dataset_available": (Path(config["data_root"]) / "idx/train_visible_1.txt").is_file()}
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        tensor = torch.randn(128, 128, device="cuda")
        result.update(gpu=device.name, compute_capability=f"{device.major}.{device.minor}",
                      total_vram_gib=device.total_memory / 1024 ** 3,
                      cuda_matmul_ok=bool(torch.isfinite(tensor @ tensor).all()))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "validate-data", "train", "evaluate"])
    parser.add_argument("--config", default="configs/regdb_laptop.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trial", type=int, choices=range(1, 11))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--feature-batch-size", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--disable-chp", action="store_true")
    parser.add_argument("--disable-khn", action="store_true")
    parser.add_argument("--disable-pts", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    for module in ("chp", "khn", "pts"):
        if getattr(args, f"disable_{module}"):
            if config.get("method") != "rahsm":
                parser.error(f"--disable-{module} requires a RAHSM config")
            config["rahsm"][module] = False
    validate_rahsm(config)
    for key in ("trial", "seed", "epochs", "iterations", "warmup_epochs", "batch_size", "feature_batch_size"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.data_root:
        config["data_root"] = str(args.data_root.resolve())
    if args.output:
        config["output_dir"] = str(args.output.resolve())
    if args.command == "doctor":
        doctor(config)
    elif args.command == "validate-data":
        report = audit_regdb(config["data_root"], check_images=args.verify_images)
        print(json.dumps(report, indent=2), flush=True)
    elif args.command == "train":
        checkpoint = train(config, resume=args.resume)
        print(f"Saved checkpoint: {checkpoint}", flush=True)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        evaluate_checkpoint(args.checkpoint, root_override=args.data_root,
                            feature_batch_size=args.feature_batch_size)


if __name__ == "__main__":
    main()
