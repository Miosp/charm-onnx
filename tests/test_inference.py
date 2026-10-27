import numpy as np
import pytest
import torch
from PIL import Image

from charm_onnx.export import load_charm
from charm_onnx.inference import CharmScorer
from charm_onnx.specs import DATASETS, PAD_SCALE_ID

PREDICTION_ATOL = 1e-4
# End to end the tokenizer may swap a high-res region (see test_tokenizer), which moves
# the score slightly. Tolerance is relative to the dataset's score span.
SCORE_TOL_FRACTION_OF_SPAN = 0.01


def reference_score(prediction: torch.Tensor, dataset: str) -> torch.Tensor:
    values = DATASETS[dataset].score_values
    return prediction[:, 0] if values is None else (prediction * torch.tensor(values)).sum(dim=-1)


def score_span(dataset: str) -> float:
    values = DATASETS[dataset].score_values
    return 1.0 if values is None else max(values) - min(values)


@pytest.fixture(scope="module", params=list(DATASETS))
def case(request, exported_model_dir, reference_tokenizer):
    dataset = request.param
    return dataset, CharmScorer(exported_model_dir(dataset), dataset), load_charm(dataset), reference_tokenizer(dataset)


def test_onnx_matches_pytorch(case, image_paths):
    dataset, scorer, charm, reference = case
    for path in image_paths:
        patches, pos_embed, scale_ids = (t.unsqueeze(0) for t in reference.preprocess(str(path)))
        with torch.no_grad():
            # The reference forward rewrites the mask in place.
            expected = charm(patches, pos_embed, scale_ids.clone())
        prediction, score = scorer.run(patches.numpy(), pos_embed.numpy(), scale_ids.numpy().astype(np.int64))
        np.testing.assert_allclose(prediction, expected.numpy(), atol=PREDICTION_ATOL, err_msg=path.name)
        np.testing.assert_allclose(score, reference_score(expected, dataset).numpy(), atol=PREDICTION_ATOL)


def test_padded_batch_matches_pytorch(case, image_paths, reference_tokenizer):
    """Padding tokens and batch > 1, on the two smallest images (larger ones would be randomly dropped)."""
    dataset, scorer, charm, _ = case
    reference = reference_tokenizer(dataset, without_pad_or_dropping=False)
    smallest = sorted(image_paths, key=lambda path: Image.open(path).width * Image.open(path).height)[:2]
    patches, pos_embed, scale_ids = (torch.stack(t) for t in zip(*(reference.preprocess(str(p)) for p in smallest)))
    assert (scale_ids == PAD_SCALE_ID).any()
    with torch.no_grad():
        expected = charm(patches, pos_embed, scale_ids.clone())
    prediction, _ = scorer.run(patches.numpy(), pos_embed.numpy(), scale_ids.numpy().astype(np.int64))
    np.testing.assert_allclose(prediction, expected.numpy(), atol=PREDICTION_ATOL)


def test_end_to_end_score_matches_reference(case, image_paths):
    dataset, scorer, charm, reference = case
    for path in image_paths:
        patches, pos_embed, scale_ids = (t.unsqueeze(0) for t in reference.preprocess(str(path)))
        with torch.no_grad():
            expected = float(reference_score(charm(patches, pos_embed, scale_ids), dataset)[0])
        tolerance = SCORE_TOL_FRACTION_OF_SPAN * score_span(dataset)
        assert scorer.score(path) == pytest.approx(expected, abs=tolerance), path.name
