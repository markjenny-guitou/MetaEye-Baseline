import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential


ROOT = Path(__file__).resolve().parents[1]


def load_upstream(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}. Run scripts/fetch_assets.py sources.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReIDEncoder(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        source = load_upstream("upstream_transreid", "third_party/TransReID/model/backbones/vit_pytorch.py")
        self.backbone = source.vit_base_patch16_224_TransReID(
            img_size=(config["height"], config["width"]), stride_size=config["stride"],
            camera=0, view=0, num_classes=0, local_feature=False, drop_path_rate=0.1)
        self.neck = nn.BatchNorm1d(768)
        self.neck.bias.requires_grad_(False)
        self.feature_dim = 768
        self.gradient_checkpointing = config["gradient_checkpointing"]
        if pretrained:
            weight_path = ROOT / config["pretrained"]
            if not weight_path.is_file():
                raise FileNotFoundError(f"Missing {weight_path}. Run scripts/fetch_assets.py weights.")
            weights = torch.load(weight_path, map_location="cpu", weights_only=True)
            weights = weights.get("model", weights.get("state_dict", weights))
            weights = {key: value for key, value in weights.items()
                       if not key.startswith(("head.", "pre_logits."))}
            weights["pos_embed"] = source.resize_pos_embed(
                weights["pos_embed"], self.backbone.pos_embed,
                self.backbone.patch_embed.num_y, self.backbone.patch_embed.num_x)
            self.backbone.load_state_dict(weights, strict=True)

    def forward(self, images):
        model = self.backbone
        if self.training and self.gradient_checkpointing:
            features = model.patch_embed(images)
            tokens = model.cls_token.expand(images.shape[0], -1, -1)
            features = model.pos_drop(torch.cat([tokens, features], dim=1) + model.pos_embed)
            features = checkpoint_sequential(model.blocks, 3, features, use_reentrant=False)
            features = model.norm(features)[:, 0]
        else:
            features = model(images)
        return F.normalize(self.neck(features).float(), dim=1)
