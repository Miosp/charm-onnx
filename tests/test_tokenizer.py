import numpy as np
import pytest
import torch

from charm_onnx import tokenizer as charm_tokenizer
from charm_onnx.export import load_base_pos_embed
from charm_onnx.specs import DATASETS
from charm_onnx.tokenizer import HIGH_RES_REGIONS, REGION_SIZE, CharmTokenizer

ATOL = 1e-5
# FFT rounding decides whether a mathematically zero coefficient becomes -inf (see
# `_sharpness`), so a region at the selection cut-off may be swapped now and then.
MAX_SWAPPED_REGIONS = 1


def make_tokenizer(dataset: str) -> CharmTokenizer:
    spec = DATASETS[dataset]
    return CharmTokenizer(load_base_pos_embed(spec), spec)


def reference_sharpness(reference, regions: np.ndarray) -> np.ndarray:
    return np.array([float(reference.calculate_blur_metric(torch.from_numpy(r))) for r in regions])


@pytest.mark.parametrize("dataset", ["aadb", "tad66k", "ava", "para"])
def test_matches_reference_given_reference_sharpness(dataset, image_paths, reference_tokenizer, monkeypatch):
    reference = reference_tokenizer(dataset)
    monkeypatch.setattr(charm_tokenizer, "_sharpness", lambda regions: reference_sharpness(reference, regions))
    tokenizer = make_tokenizer(dataset)
    for path in image_paths:
        patches, pos_embed, scale_ids = tokenizer(path)
        ref_patches, ref_pos_embed, ref_scale_ids = (t.numpy() for t in reference.preprocess(str(path)))
        assert patches.shape == ref_patches.shape, path.name
        np.testing.assert_array_equal(scale_ids, ref_scale_ids.astype(np.int64), err_msg=path.name)
        np.testing.assert_allclose(patches, ref_patches, atol=ATOL, err_msg=path.name)
        np.testing.assert_allclose(pos_embed, ref_pos_embed, atol=ATOL, err_msg=path.name)


def test_selects_reference_high_res_regions(image_paths, reference_tokenizer):
    reference = reference_tokenizer("aadb")
    tokenizer = make_tokenizer("aadb")
    for path in image_paths:
        image = charm_tokenizer._pad_or_crop(tokenizer.load_image(path), REGION_SIZE)
        regions = charm_tokenizer._split_grid(image, REGION_SIZE)
        if len(regions) < HIGH_RES_REGIONS:
            continue
        ours = np.argsort(charm_tokenizer._sharpness(regions), kind="stable")[-HIGH_RES_REGIONS:]
        theirs = np.argsort(reference_sharpness(reference, regions), kind="stable")[-HIGH_RES_REGIONS:]
        assert len(np.setdiff1d(ours, theirs)) <= MAX_SWAPPED_REGIONS, path.name


def test_both_scales_exercised(image_paths):
    tokenizer = make_tokenizer("aadb")
    multi_scale = [bool(tokenizer(path)[2].any()) for path in image_paths]
    assert any(multi_scale) and not all(multi_scale)
