import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .clustering import aggregate_memories, cluster_features
from .data import TrainingImages, PseudoIdentityBatchSampler, read_split, audit_regdb
from .evaluation import extract_features
from .model import ReIDEncoder
from .artifacts import capture_run, new_archive, retain_file, write_json, write_npz
from .rahsm import RAHSMBank, associate_shared_labels, iteration_lambda, validate_rahsm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def split_fingerprint(root, trial):
    root = Path(root)
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for modal in ("visible", "thermal"):
            manifest = root / "idx" / f"{split}_{modal}_{trial}.txt"
            digest.update(manifest.read_bytes())
    return digest.hexdigest()


def save_checkpoint(path, state):
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def update_memory(memory, features, targets, momentum):
    # Sequential prototype updates match ADCA's ClusterMemory (non-hard mode).
    for feature, target in zip(features.detach().float(), targets):
        memory[target].mul_(momentum).add_(feature, alpha=1 - momentum)
        memory[target] = F.normalize(memory[target], dim=0)


def train(config, resume=None):
    validate_rahsm(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. Run main.py doctor before training.")
    if config["epochs"] < 1 or config["iterations"] < 1:
        raise ValueError("epochs and iterations must be positive")
    if not 0 <= config["warmup_epochs"] <= config["epochs"]:
        raise ValueError("warmup_epochs must be between zero and epochs")
    audit_regdb(config["data_root"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint_last.pt").exists() and not resume:
        raise FileExistsError(f"{output} already has a checkpoint. Select another output or use --resume.")
    fingerprint = split_fingerprint(config["data_root"], config["trial"])
    device = torch.device("cuda")
    set_seed(config["seed"])
    torch.set_num_threads(4)
    model = ReIDEncoder(config, pretrained=resume is None).to(device)
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad),
                                lr=config["learning_rate"], momentum=0.9, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    start_epoch = 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        for key in config:
            if key != "output_dir" and state["config"].get(key) != config[key]:
                raise ValueError(f"Resume config differs for {key}")
        if state["split_sha256"] != fingerprint:
            raise ValueError("RegDB split manifests changed since the checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = {modal: read_split(config["data_root"], "train", modal, config["trial"])
               for modal in ("visible", "thermal")}
    provenance = capture_run(output, config, records)
    print(f"Train images: visible={len(records['visible'])}, thermal={len(records['thermal'])}. "
          f"Resume epoch={start_epoch}; ground-truth identities are excluded from optimization.", flush=True)
    for epoch in range(start_epoch, config["epochs"]):
        set_seed(config["seed"] + epoch)
        started = time.perf_counter()
        archive = new_archive(output / "epochs", f"epoch_{epoch + 1:04d}")
        write_json(archive / "summary.json", {"epoch": epoch + 1, "status": "started",
                                              "provenance": str(provenance)})
        learning_rate = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats()
        features = {modal: extract_features(model, values, config, device)
                    for modal, values in records.items()}
        labels, centers = {}, {}
        for modal in records:
            labels[modal], centers[modal] = cluster_features(features[modal], config[f"eps_{modal}"], config)
        write_npz(archive / "clustering.npz",
                  **{f"labels_{modal}": labels[modal] for modal in records},
                  **{f"features_{modal}": features[modal].numpy() for modal in records},
                  **{f"centers_{modal}": centers[modal].numpy() for modal in records})
        matches = []
        rahsm = None
        if config.get("method") == "rahsm" and epoch >= config["warmup_epochs"]:
            settings = config["rahsm"]
            labels, association = associate_shared_labels(
                features, labels, settings["cross_neighbors"], settings["smoothing_neighbors"])
            rahsm = RAHSMBank.from_features(features, labels, config, config["seed"] + epoch).to(device)
            centers = {"visible": rahsm.centers[0], "thermal": rahsm.centers[1]}
            write_json(archive / "shared_label_association.json", association)
            write_npz(archive / "shared_labels.npz", **labels)
            write_npz(archive / "rahsm_initial_bank.npz", **rahsm.archive_arrays())
        elif config.get("method", "adca_transreid") == "adca_transreid" and epoch >= config["warmup_epochs"]:
            centers["visible"], centers["thermal"], matches = aggregate_memories(
                features["visible"], features["thermal"], labels["visible"], labels["thermal"],
                centers["visible"], centers["thermal"], config["cma_threshold"], config["cma_weight"])
        write_json(archive / "cma_pairs.json", matches)
        preparation_seconds = time.perf_counter() - started
        cluster_summary = {modal: {"clusters": len(centers[modal]), "outliers": int(sum(labels[modal] < 0))}
                           for modal in records}
        print(f"Epoch {epoch + 1}/{config['epochs']} clusters={cluster_summary} CMA_pairs={len(matches)}", flush=True)
        loaders = {}
        for index, modal in enumerate(records):
            dataset = TrainingImages(records[modal], labels[modal], config["height"], config["width"], visible=modal == "visible")
            sampler = PseudoIdentityBatchSampler(labels[modal], config["batch_size"], config["instances"],
                                                config["iterations"], config["seed"] + epoch * 2 + index)
            loaders[modal] = DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"], pin_memory=True)
        memory = {modal: centers[modal].to(device) for modal in records}
        model.train()
        loss_sum = 0.0
        skipped_steps = 0
        for step, (visible, thermal) in enumerate(zip(loaders["visible"], loaders["thermal"]), 1):
            magnitude = iteration_lambda(config, epoch, step) if rahsm is not None else 0.0
            images_rgb, images_channel, labels_rgb = (tensor.to(device, non_blocking=True) for tensor in visible)
            images_ir, labels_ir = (tensor.to(device, non_blocking=True) for tensor in thermal)
            visible_targets = torch.cat([labels_rgb, labels_rgb])
            images = torch.cat([images_rgb, images_channel, images_ir])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=config["amp"]):
                embeddings = model(images)
            embeddings_rgb = embeddings[:len(visible_targets)]
            embeddings_ir = embeddings[len(visible_targets):]
            # Snapshot the memories before backward, then update only after a valid optimizer step.
            if rahsm is not None:
                negatives = rahsm.negative_candidates(magnitude) if config["rahsm"]["khn"] else None
                loss_rgb, roles_rgb = rahsm.loss(embeddings_rgb, visible_targets, 0, magnitude, negatives)
                loss_ir, roles_ir = rahsm.loss(embeddings_ir, labels_ir, 1, magnitude, negatives)
                if step == config["iterations"]:
                    arrays = {"centers_before_update": rahsm.centers.detach().cpu().numpy(),
                              "lambda": np.asarray(magnitude)}
                    for modal, queries, targets, roles in (
                            ("visible", embeddings_rgb, visible_targets, roles_rgb),
                            ("thermal", embeddings_ir, labels_ir, roles_ir)):
                        arrays[f"queries_{modal}"] = queries.detach().cpu().numpy()
                        arrays[f"targets_{modal}"] = targets.detach().cpu().numpy()
                        arrays.update({f"{name}_{modal}": value.cpu().numpy() for name, value in roles.items()})
                    if negatives is not None:
                        arrays["negative_neighbors"] = negatives[2].cpu().numpy()
                    write_npz(archive / "rahsm_last_batch_roles.npz", **arrays)
            else:
                loss_rgb = F.cross_entropy(embeddings_rgb @ memory["visible"].T / config["temperature"], visible_targets)
                loss_ir = F.cross_entropy(embeddings_ir @ memory["thermal"].T / config["temperature"], labels_ir)
            loss = loss_rgb + loss_ir
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                if rahsm is not None:
                    rahsm.update(embeddings_rgb, visible_targets, embeddings_ir, labels_ir)
                else:
                    update_memory(memory["visible"], embeddings_rgb, visible_targets, config["memory_momentum"])
                    update_memory(memory["thermal"], embeddings_ir, labels_ir, config["memory_momentum"])
            else:
                skipped_steps += 1
            with (archive / "steps.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "loss": float(loss.detach()),
                    "loss_visible": float(loss_rgb.detach()), "loss_thermal": float(loss_ir.detach()),
                    "learning_rate": learning_rate, "amp_scale": scaler.get_scale(),
                    "optimizer_updated": scaler.get_scale() >= old_scale,
                    "rahsm_active": rahsm is not None, "lambda": magnitude}) + "\n")
            loss_sum += float(loss.detach())
            if step == 1 or step % 10 == 0 or step == config["iterations"]:
                print(f"  step {step}/{config['iterations']} loss={float(loss.detach()):.5f}", flush=True)
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": loss_sum / config["iterations"], "clusters": cluster_summary,
               "cma_pairs": len(matches), "seconds": time.perf_counter() - started,
               "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
               "skipped_steps": skipped_steps, "learning_rate": learning_rate,
               "next_learning_rate": optimizer.param_groups[0]["lr"],
               "preparation_seconds": preparation_seconds,
               "method": config.get("method", "adca_transreid"), "rahsm_active": rahsm is not None,
               "lambda_last": magnitude,
               "archive": str(archive), "provenance": str(provenance)}
        if rahsm is not None:
            write_npz(archive / "rahsm_final_centers.npz", centers=rahsm.centers.detach().cpu().numpy())
        write_npz(output / "pseudo_labels_latest.npz", **labels)
        write_json(output / "cma_pairs_latest.json", matches)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "config": config, "epoch": epoch + 1, "split_sha256": fingerprint, "epoch_summary": row}
        save_checkpoint(output / "checkpoint_last.pt", state)
        if epoch + 1 in {config["warmup_epochs"], config["epochs"]}:
            retain_file(output / "checkpoint_last.pt", archive / "checkpoint.pt")
        write_json(archive / "summary.json", {**row, "status": "completed",
            "clustering_model_epoch": epoch, "checkpoint_model_epoch": epoch + 1})
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return output / "checkpoint_last.pt"
