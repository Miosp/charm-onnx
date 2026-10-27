"""Per-dataset settings of the published Charm checkpoints (mirrors Charm_tokenizer 1.0.17)."""

from dataclasses import dataclass

HF_REPO_ID = "FatemehBehrad/Charm"  # original PyTorch checkpoints
HF_ONNX_REPO_ID = "Miosp/charm-onnx"  # exported models, one directory per dataset

# Files written per dataset by `charm_onnx.export`, read by `charm_onnx.inference`.
MODEL_FILE = "model.onnx"
POS_EMBED_FILE = "pos_embed.npy"
INPUT_NAMES = ("patches", "pos_embed", "scale_ids")
OUTPUT_NAMES = ("prediction", "score")

PATCH_SIZE = 14
PAD_SCALE_ID = 9  # scale id the tokenizer uses for padding tokens

DINO_SMALL_POS_EMBED = "dino_small_pos.pt"
DINO_LARGE_POS_EMBED = "dino_large.pt"


@dataclass(frozen=True)
class DatasetSpec:
    checkpoint: str
    pos_embed_file: str
    max_tokens: int  # including the CLS slot
    score_values: tuple[float, ...] | None = None  # bins of a score distribution head
    max_image_side: int | None = None  # images are downscaled to fit this before tokenizing


_SCORES_1_TO_10 = tuple(float(v) for v in range(1, 11))
_SCORES_1_TO_5_HALF_STEPS = tuple(1 + 0.5 * i for i in range(9))

DATASETS: dict[str, DatasetSpec] = {
    "aadb": DatasetSpec("aadb_charm.pth", DINO_SMALL_POS_EMBED, 1025),
    "ava": DatasetSpec("Ava_large_charm.pth", DINO_LARGE_POS_EMBED, 513, score_values=_SCORES_1_TO_10),
    "baid": DatasetSpec("baid_charm.pth", DINO_SMALL_POS_EMBED, 1025),
    "koniq10k": DatasetSpec("koniq10k_charm.pth", DINO_SMALL_POS_EMBED, 1025),
    "para": DatasetSpec(
        "para_charm.pth", DINO_SMALL_POS_EMBED, 1025, score_values=_SCORES_1_TO_5_HALF_STEPS, max_image_side=1024
    ),
    "spaq": DatasetSpec("spaq_charm.pth", DINO_SMALL_POS_EMBED, 1025, max_image_side=1024),
    "tad66k": DatasetSpec("tad66k_charm.pth", DINO_SMALL_POS_EMBED, 769),
}
