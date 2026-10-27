import warnings
from pathlib import Path

import pytest
from PIL import Image

IMAGES_DIR = Path(__file__).parent / "images"

# (source image, target size or None to keep). Sizes cover the single-scale path, both
# branches of the pad-or-crop rule, odd sizes, portrait/landscape and the >1024 px resize.
IMAGE_CASES = [
    ("chelsea", None),  # 451x300 -> single scale for every dataset but ava
    ("coffee", None),
    ("astronaut", None),
    ("rocket", None),
    ("hubble_deep_field", None),
    ("chelsea", (213, 160)),
    ("rocket", (1999, 1333)),
    ("coffee", (601, 1203)),
    ("astronaut", (1050, 1050)),
]


@pytest.fixture(scope="session")
def image_paths(tmp_path_factory) -> list[Path]:
    out_dir = tmp_path_factory.mktemp("images")
    paths = []
    for name, size in IMAGE_CASES:
        source = IMAGES_DIR / f"{name}.png"
        if size is None:
            paths.append(source)
            continue
        target = out_dir / f"{name}_{size[0]}x{size[1]}.png"
        Image.open(source).convert("RGB").resize(size, Image.Resampling.LANCZOS).save(target)
        paths.append(target)
    return paths


@pytest.fixture(scope="session")
def reference_tokenizer():
    """Factory for the upstream torch tokenizer, cached per dataset."""
    with warnings.catch_warnings():  # the module mutes all warnings on import
        from Charm_tokenizer.ImageProcessor import Charm_Tokenizer

    cache = {}

    def make(dataset: str, without_pad_or_dropping: bool = True):
        key = (dataset, without_pad_or_dropping)
        if key not in cache:
            backbone = "facebook/dinov2-large" if dataset == "ava" else "facebook/dinov2-small"
            cache[key] = Charm_Tokenizer(
                patch_selection="frequency",
                training_dataset=dataset,
                backbone=backbone,
                without_pad_or_dropping=without_pad_or_dropping,
            )
        return cache[key]

    return make


@pytest.fixture(scope="session")
def exported_model_dir(tmp_path_factory):
    """Factory exporting a dataset's model into a temp dir, so tests never see stale artifacts."""
    from charm_onnx.export import export_dataset

    out_dir = tmp_path_factory.mktemp("models")
    return lambda dataset: export_dataset(dataset, out_dir)
