---
license: apache-2.0
base_model: FatemehBehrad/Charm
library_name: onnx
pipeline_tag: image-classification
tags:
  - onnx
  - onnxruntime
  - image-aesthetics
  - image-quality-assessment
  - dinov2
---

# Charm (ONNX)

ONNX exports of the [Charm](https://github.com/FBehrad/Charm) (CVPR 2025) image
aesthetic and quality assessment models, converted from
[FatemehBehrad/Charm](https://huggingface.co/FatemehBehrad/Charm). Charm is a
tokenization scheme for ViTs that keeps composition, high resolution, aspect ratio and
multi-scale information; the backbone is DINOv2.

Code, export script and tests: https://github.com/Miosp/charm-onnx

## Usage

```sh
pip install git+https://github.com/Miosp/charm-onnx
python -m charm_onnx.inference --dataset aadb photo.jpg
```

```python
from charm_onnx.inference import CharmScorer

scorer = CharmScorer.from_pretrained("aadb")  # downloads aadb/ from this repo
scorer.score("photo.jpg")
```

Inference needs only `onnxruntime`, `numpy` and `pillow`; the Charm tokenizer is
reimplemented in NumPy.

## Files

One directory per training dataset, each with `model.onnx` (fp32, ONNX Runtime
transformer-fused) and `pos_embed.npy` (DINOv2 positional table used by the tokenizer).

| Dataset | Task | Backbone | Score |
|---|---|---|---|
| `aadb` | aesthetics | DINOv2-small | head output |
| `ava` | aesthetics | DINOv2-large | mean of 1-10 distribution |
| `baid` | aesthetics | DINOv2-small | head output |
| `koniq10k` | quality | DINOv2-small | head output |
| `para` | aesthetics | DINOv2-small | mean of 1-5 distribution |
| `spaq` | quality | DINOv2-small | head output |
| `tad66k` | aesthetics | DINOv2-small | head output |

Scores are what the upstream `Charm_tokenizer.Backbone.predict` returns.

Model inputs (slot 0 of the token axis is the CLS placeholder):

| Name | Shape | Type |
|---|---|---|
| `patches` | `[batch, tokens, 3, 14, 14]` | float32 |
| `pos_embed` | `[batch, tokens, dim]` | float32 |
| `scale_ids` | `[batch, tokens]` | int64 (0 low-res, 1 high-res, 9 padding) |

Outputs: `prediction [batch, classes]`, `score [batch]`. There is no attention mask, so
batch only images with the same token count.

## Accuracy vs. the original PyTorch models

- Same tokens in: outputs match to < 1e-6.
- Image in, end to end: scores within 0.5% of the score range on the test images. The
  difference comes from the tokenizer's FFT-based patch ranking, which is sensitive to
  floating-point rounding; occasionally one of 64 high-resolution regions differs.
- Only the `frequency` patch selection (upstream's recommended inference setting) is
  implemented.

## License and citation

Apache-2.0, as the original code and checkpoints. Changes: forward pass rewritten with
traceable ops, DINOv2 LayerScale folded into the preceding linear layers (numerically
equivalent), ONNX Runtime transformer fusions applied.

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
