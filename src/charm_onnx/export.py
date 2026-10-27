"""Convert the published Charm PyTorch checkpoints to optimized ONNX models.

Requires the `export` dependency group (torch, transformers==4.36.0, Charm-tokenizer).
"""

import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from charm_onnx.specs import (
    DATASETS,
    HF_REPO_ID,
    INPUT_NAMES,
    MODEL_FILE,
    OUTPUT_NAMES,
    PAD_SCALE_ID,
    PATCH_SIZE,
    POS_EMBED_FILE,
    DatasetSpec,
)

OPSET = 18


@torch.no_grad()
def fold_layer_scale(linear: nn.Linear, gamma: torch.Tensor) -> None:
    """Merge a DINOv2 LayerScale into the preceding Linear; the extra Mul blocks ORT attention fusion."""
    linear.weight.mul_(gamma.unsqueeze(1))
    linear.bias.mul_(gamma)


class ExportableCharm(nn.Module):
    """Charm forward pass rewritten with traceable ops.

    Inputs follow the Charm tokenizer output (slot 0 is the CLS placeholder):
      patches   [batch, tokens, 3, 14, 14] float32
      pos_embed [batch, tokens, dim]       float32
      scale_ids [batch, tokens]            int64 (0..scales-1, or PAD_SCALE_ID for padding)
    """

    def __init__(self, charm: nn.Module, spec: DatasetSpec):
        super().__init__()
        charm = copy.deepcopy(charm)
        embeddings = charm.patch_embeddings
        conv = embeddings.patch_embedder
        dim = conv.out_channels
        # A stride-14 conv over a 14x14 patch is a single dot product per output channel.
        self.patch_proj = nn.Linear(conv.in_channels * PATCH_SIZE * PATCH_SIZE, dim)
        self.patch_proj.weight = nn.Parameter(conv.weight.detach().reshape(dim, -1).clone())
        self.patch_proj.bias = nn.Parameter(conv.bias.detach().clone())
        self.cls_token = embeddings.cls_token
        self.mask_token = embeddings.mask_token
        self.scale_embed = nn.Parameter(embeddings.se.detach()[0].clone())
        self.encoder = charm.encoder.encoder
        for layer in self.encoder.layer:
            fold_layer_scale(layer.attention.output.dense, layer.layer_scale1.lambda1)
            fold_layer_scale(layer.mlp.fc2, layer.layer_scale2.lambda1)
            layer.layer_scale1 = layer.layer_scale2 = nn.Identity()
        self.layernorm = charm.encoder.layernorm
        self.head = charm.mlp_head.linear
        self.head_activation = charm.mlp_head.activation
        score_values = torch.tensor(spec.score_values) if spec.score_values else None
        self.register_buffer("score_values", score_values)

    def forward(self, patches, pos_embed, scale_ids):
        batch, tokens = patches.shape[:2]
        x = self.patch_proj(patches[:, 1:].reshape(batch, tokens - 1, -1))
        scale_ids = scale_ids[:, 1:]
        is_pad = scale_ids == PAD_SCALE_ID
        x = torch.where(is_pad.unsqueeze(-1), self.mask_token, x)
        x = x + self.scale_embed[torch.where(is_pad, 0, scale_ids)]
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1) + pos_embed

        hidden = self.layernorm(self.encoder(x).last_hidden_state)
        pooled = torch.cat((hidden[:, 0], hidden[:, 1:].mean(dim=1)), dim=1)
        prediction = self.head_activation(self.head(pooled))
        if self.score_values is None:
            score = prediction[:, 0]
        else:
            score = (prediction * self.score_values).sum(dim=-1)
        return prediction, score


def load_charm(dataset: str) -> nn.Module:
    # Importing Backbone puts the Charm_tokenizer dir on sys.path, which the pickled `model.Model` needs.
    # catch_warnings undoes the global "ignore all" filter the unpickled module installs on import.
    with warnings.catch_warnings():
        from Charm_tokenizer.Backbone import backbone

        return backbone(training_dataset=dataset, device="cpu").model.eval()


def load_base_pos_embed(spec: DatasetSpec) -> np.ndarray:
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=spec.pos_embed_file)
    return torch.load(path, map_location="cpu").detach().float().numpy()


def sample_inputs(dim: int, tokens: int = 65) -> tuple[torch.Tensor, ...]:
    patches = torch.rand(1, tokens, 3, PATCH_SIZE, PATCH_SIZE)
    pos_embed = torch.randn(1, tokens, dim) * 0.02
    scale_ids = torch.randint(0, 2, (1, tokens), dtype=torch.int64)
    return patches, pos_embed, scale_ids


def export_onnx(module: ExportableCharm, path: Path) -> None:
    dynamic_axes = {
        "patches": {0: "batch", 1: "tokens"},
        "pos_embed": {0: "batch", 1: "tokens"},
        "scale_ids": {0: "batch", 1: "tokens"},
        "prediction": {0: "batch"},
        "score": {0: "batch"},
    }
    with warnings.catch_warnings():
        # transformers 4.36 is noisy under current torch; TracerWarnings stay visible.
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            module,
            sample_inputs(module.patch_proj.out_features),
            str(path),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )


def optimize_onnx(path: Path, num_heads: int, hidden_size: int) -> None:
    """Fuse attention / LayerNorm / GELU into ONNX Runtime contrib ops, in place."""
    from onnxruntime.transformers import optimizer

    optimized = optimizer.optimize_model(
        str(path), model_type="vit", num_heads=num_heads, hidden_size=hidden_size, opt_level=0
    )
    print(f"  fused ops: {optimized.get_fused_operator_statistics()}")
    optimized.save_model_to_file(str(path))


def export_dataset(dataset: str, out_dir: Path) -> Path:
    spec = DATASETS[dataset]
    target = out_dir / dataset
    target.mkdir(parents=True, exist_ok=True)

    charm = load_charm(dataset)
    module = ExportableCharm(charm, spec).eval()
    config = module.encoder.config
    model_path = target / MODEL_FILE
    with torch.no_grad():
        export_onnx(module, model_path)
    optimize_onnx(model_path, config.num_attention_heads, config.hidden_size)
    np.save(target / POS_EMBED_FILE, load_base_pos_embed(spec))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=list(DATASETS), help="default: all")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    for dataset in args.datasets or DATASETS:
        print(f"exporting {dataset}")
        print(f"  -> {export_dataset(dataset, args.out_dir)}")


if __name__ == "__main__":
    main()
