"""NumPy port of the Charm tokenizer (Charm_tokenizer 1.0.17, `ImageProcessor.Charm_Tokenizer`).

Covers the configuration recommended for inference: frequency-based patch selection,
2 scales, downscaling factor 0.5, no padding or dropping. Quirks of the reference
implementation (pad-or-crop rule, positional-embedding order of high-resolution
tokens) are reproduced on purpose because the checkpoints were trained with them.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from charm_onnx.specs import PATCH_SIZE, DatasetSpec

REGION_SIZE = 2 * PATCH_SIZE  # a high-res region is 2x2 tokens, a low-res region 1 token
HIGH_RES_REGIONS = 64  # int(512 * (1 / scales) / (2 ** (scales - 1)) ** 2) in the reference
POS_EMBED_INTERPOLATE_OFFSET = 0.1
BICUBIC_A = -0.75  # PyTorch's bicubic kernel coefficient
LOW_RES, HIGH_RES = 0, 1  # scale ids


def _cubic_weights(t: np.ndarray) -> np.ndarray:
    """Weights of the 4 neighbours at offsets -1..2 for fractional positions t."""

    def near(x):
        return ((BICUBIC_A + 2) * x - (BICUBIC_A + 3)) * x * x + 1

    def far(x):
        return ((BICUBIC_A * x - 5 * BICUBIC_A) * x + 8 * BICUBIC_A) * x - 4 * BICUBIC_A

    return np.stack([far(t + 1), near(t), near(1 - t), far(2 - t)], axis=-1)


def _bicubic_axis(x: np.ndarray, axis: int, out_size: int, src_per_dst: float) -> np.ndarray:
    """1-D bicubic resampling matching torch `interpolate(mode="bicubic", align_corners=False)`."""
    x = np.moveaxis(x, axis, -1)
    src = (np.arange(out_size) + 0.5) * src_per_dst - 0.5
    base = np.floor(src)
    neighbours = np.clip(base.astype(np.int64)[:, None] + np.arange(-1, 3), 0, x.shape[-1] - 1)
    resampled = (x[..., neighbours] * _cubic_weights(src - base)).sum(axis=-1)
    return np.moveaxis(resampled, -1, axis)


def bicubic_resize(x: np.ndarray, out_hw: tuple[int, int], src_per_dst: tuple[float, float]) -> np.ndarray:
    """Resize the last two axes of x; src_per_dst is the coordinate scale PyTorch uses per axis."""
    x = _bicubic_axis(x.astype(np.float64), -1, out_hw[1], src_per_dst[1])
    return _bicubic_axis(x, -2, out_hw[0], src_per_dst[0]).astype(np.float32)


def _pad_bottom_right(image: np.ndarray, multiple: int) -> np.ndarray:
    _, height, width = image.shape
    return np.pad(image, ((0, 0), (0, -height % multiple), (0, -width % multiple)))


def _pad_or_crop(image: np.ndarray, multiple: int) -> np.ndarray:
    """Pad or crop to a multiple of `multiple`, whichever the reference's area heuristic picks."""
    _, height, width = image.shape
    crop_area = max(height % multiple, 1) * max(width % multiple, 1)
    pad_area = (-height % multiple or 1) * (-width % multiple or 1)
    if pad_area > crop_area:
        return image[:, : height - height % multiple, : width - width % multiple]
    return _pad_bottom_right(image, multiple)


def _split_grid(image: np.ndarray, size: int) -> np.ndarray:
    """(C, H, W) -> (rows * cols, C, size, size) in raster order."""
    channels, height, width = image.shape
    rows, cols = height // size, width // size
    grid = image.reshape(channels, rows, size, cols, size).transpose(1, 3, 0, 2, 4)
    return grid.reshape(rows * cols, channels, size, size)


def _sharpness(regions: np.ndarray) -> np.ndarray:
    """Mean log-magnitude spectrum per region (the reference's FFT blur metric).

    A spectrum coefficient that rounds to exactly 0 makes the metric -inf. Whether that
    happens for a mathematically zero coefficient depends on the FFT implementation's
    rounding, so rarely a region near the selection cut-off differs from PyTorch's choice.
    """
    with np.errstate(divide="ignore"):
        magnitude = 20 * np.log10(np.abs(np.fft.fft2(regions.astype(np.float64))))
    return magnitude.mean(axis=(1, 2, 3))


class CharmTokenizer:
    def __init__(self, base_pos_embed: np.ndarray, spec: DatasetSpec):
        """base_pos_embed: [1, 1 + side * side, dim] positional table of the DINOv2 backbone."""
        self.spec = spec
        self.cls_pos_embed = base_pos_embed[0, :1].astype(np.float32)
        patch_pos = base_pos_embed[0, 1:]
        side = int(round(np.sqrt(patch_pos.shape[0])))
        self.pos_grid = patch_pos.reshape(side, side, -1).transpose(2, 0, 1)  # (dim, side, side)
        self._pos_cache: dict[tuple[int, int], np.ndarray] = {}

    def grid_pos_embed(self, rows: int, cols: int) -> np.ndarray:
        """Positional embeddings interpolated to a rows x cols token grid, raster order."""
        key = (rows, cols)
        if key not in self._pos_cache:
            side = self.pos_grid.shape[-1]
            # The reference passes scale_factor=(n + offset) / side, so torch maps coordinates by its inverse.
            src_per_dst = (side / (rows + POS_EMBED_INTERPOLATE_OFFSET), side / (cols + POS_EMBED_INTERPOLATE_OFFSET))
            grid = bicubic_resize(self.pos_grid, key, src_per_dst)
            self._pos_cache[key] = grid.reshape(grid.shape[0], -1).T
        return self._pos_cache[key]

    def load_image(self, path: str | Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        max_side = self.spec.max_image_side
        if max_side is not None and max(image.size) > max_side:
            ratio = min(max_side / image.width, max_side / image.height)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)))
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255

    def __call__(self, path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (patches [N, 3, 14, 14] f32, pos_embed [N, dim] f32, scale_ids [N] i64); slot 0 is CLS."""
        return self.tokenize(self.load_image(path))

    def tokenize(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """image: [3, H, W] float32 in [0, 1]."""
        _, height, width = image.shape
        if -(-height // PATCH_SIZE) * -(-width // PATCH_SIZE) <= self.spec.max_tokens:
            patches, pos_embed, scale_ids = self._single_scale(image)
        else:
            patches, pos_embed, scale_ids = self._multi_scale(image)
        cls_patch = np.zeros((1, *patches.shape[1:]), dtype=np.float32)
        return (
            np.concatenate([cls_patch, patches]),
            np.concatenate([self.cls_pos_embed, pos_embed]),
            np.concatenate([[LOW_RES], scale_ids]).astype(np.int64),
        )

    def _single_scale(self, image: np.ndarray):
        image = _pad_bottom_right(image, PATCH_SIZE)
        _, height, width = image.shape
        patches = _split_grid(image, PATCH_SIZE)
        pos_embed = self.grid_pos_embed(height // PATCH_SIZE, width // PATCH_SIZE)
        return patches, pos_embed, np.full(len(patches), LOW_RES)

    def _multi_scale(self, image: np.ndarray):
        image = _pad_or_crop(image, REGION_SIZE)
        _, height, width = image.shape
        rows, cols = height // REGION_SIZE, width // REGION_SIZE
        regions = _split_grid(image, REGION_SIZE)
        if len(regions) < HIGH_RES_REGIONS:
            raise ValueError(f"image has {len(regions)} regions, fewer than {HIGH_RES_REGIONS}")

        # Stable sort keeps the reference's tie order (e.g. between all-zero padding regions).
        ranked = np.argsort(_sharpness(regions), kind="stable")
        high = np.sort(ranked[-HIGH_RES_REGIONS:])
        low = np.setdiff1d(np.arange(len(regions)), high)

        low_patches = bicubic_resize(regions[low], (PATCH_SIZE, PATCH_SIZE), (2.0, 2.0))
        high_patches = (
            regions[high]
            .reshape(len(high), 3, 2, PATCH_SIZE, 2, PATCH_SIZE)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
        )

        # High-res positions are taken in raster order of the fine grid, not in token order
        # (region by region); the reference does the same and the checkpoints depend on it.
        fine_mask = np.zeros((rows, cols), dtype=bool)
        fine_mask.flat[high] = True
        fine_mask = fine_mask.repeat(2, axis=0).repeat(2, axis=1)
        pos_embed = np.concatenate(
            [self.grid_pos_embed(rows, cols)[low], self.grid_pos_embed(2 * rows, 2 * cols)[fine_mask.ravel()]]
        )
        scale_ids = np.repeat([LOW_RES, HIGH_RES], [len(low_patches), len(high_patches)])
        return np.concatenate([low_patches, high_patches]), pos_embed, scale_ids
