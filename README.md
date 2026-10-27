# charm-onnx

ONNX Runtime inference for [Charm](https://github.com/FBehrad/Charm) (CVPR 2025) image
aesthetic / quality assessment models. Inference needs only `onnxruntime`, `numpy` and
`pillow`; PyTorch is used once, to convert the published checkpoints.

## Usage

```sh
uv sync                                   # Python 3.12, includes the export group
uv run python -m charm_onnx.export        # all datasets -> models/<dataset>/
uv run python -m charm_onnx.export aadb   # or just some
uv run python -m charm_onnx.inference --dataset aadb photo.jpg
```

```python
from charm_onnx.inference import CharmScorer

scorer = CharmScorer("models/aadb", "aadb")
scorer.score("photo.jpg")
```

Datasets: `aadb`, `ava` (DINOv2-large), `baid`, `koniq10k`, `para`, `spaq`, `tad66k`.
Scores are what upstream `backbone.predict` returns (distribution mean for `ava`/`para`,
the head output otherwise).

For deployment copy `models/<dataset>/` (`model.onnx`, `pos_embed.npy`) and install the
package without the export group: `uv sync --no-default-groups`.

## What the export does

- Rewrites the forward pass with traceable ops: the scale-embedding lookup upstream does
  in NumPy becomes a gather, the in-place mask edit becomes `where`, the 14x14 patch
  conv becomes a `Linear`.
- Folds DINOv2 LayerScale into the preceding projections. The extra `Mul` otherwise
  stops ONNX Runtime from fusing attention.
- Runs the ONNX Runtime transformer optimizer: every layer ends up as fused
  `Attention`, `BiasGelu` and `SkipLayerNormalization` contrib ops.
- Dynamic `batch` and `tokens` axes. The model has no attention mask, so only batch
  sequences of equal length; padding changes the score.

`charm_onnx.tokenizer` is a NumPy port of `Charm_tokenizer.Charm_Tokenizer` for the
recommended inference setting (frequency patch selection, 2 scales, no pad/drop).

## Accuracy (tests/, vs. upstream Charm-tokenizer 1.0.17 on CPU)

- ONNX model vs. PyTorch on identical tokens: max abs difference < 1e-6 (test bound 1e-4).
- Tokenizer, given the same patch ranking: identical tokens, positions and scale ids.
- End to end: score within 0.5% of the score span on the test images (test bound 1%).
  The gap comes from upstream's FFT sharpness metric: it takes `log10` of the
  spectrum, so a coefficient that rounds to exactly 0 ranks a region as blurriest.
  Whether a mathematically zero coefficient rounds to 0 depends on the FFT
  implementation, so now and then one of the 64 high-resolution regions differs.

`uv run pytest` runs everything (~12 min on CPU; exports all models into a temp dir first).

## Performance

i7-1260P laptop, CPU, fp32, single image: ONNX Runtime is 1.2-1.8x faster than eager
PyTorch (e.g. 517 tokens: ~125 ms vs ~180 ms). Tokenizing takes 10-200 ms depending on
image size. Except for `para`/`spaq` (capped at 1024 px), images are not downscaled, so
token count, memory and time grow with resolution, as upstream.

int8 dynamic quantization was tried and rejected: ~1.4x faster, but scores moved by up to
0.024 (aadb, 0-1 scale) and 0.41 (tad66k).

## Upstream issues found

- `Charm_tokenizer` fails on Python 3.13+ (`KeyError: 'selected_indices_l1'`): it
  assigns through `locals()`, which PEP 667 made a snapshot. Hence Python 3.12 here.
- The `.pth` checkpoints are pickled model objects, not state dicts; loading them needs
  `transformers==4.36.0` and `ml-collections`.
- High-resolution tokens are paired with positional embeddings in fine-grid raster order,
  not their own (region-by-region) order. Reproduced, since the checkpoints were
  trained that way.

## License and citation

Apache-2.0, like the original Charm code and checkpoints; see `LICENSE` and `NOTICE`.
If you use the models, cite the Charm paper:

```bibtex
@InProceedings{Behrad_2025_CVPR,
    author    = {Behrad, Fatemeh and Tuytelaars, Tinne and Wagemans, Johan},
    title     = {Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {7815-7824}
}
```
