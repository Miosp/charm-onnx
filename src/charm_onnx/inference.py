"""Score images with an exported Charm model using only ONNX Runtime and NumPy."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from charm_onnx.specs import DATASETS, INPUT_NAMES, MODEL_FILE, POS_EMBED_FILE
from charm_onnx.tokenizer import CharmTokenizer


class CharmScorer:
    def __init__(
        self,
        model_dir: str | Path,
        dataset: str,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ):
        """model_dir: directory written by `charm_onnx.export` for this dataset."""
        model_dir = Path(model_dir)
        self.tokenizer = CharmTokenizer(np.load(model_dir / POS_EMBED_FILE), DATASETS[dataset])
        self.session = ort.InferenceSession(
            model_dir / MODEL_FILE, sess_options=session_options, providers=providers or ["CPUExecutionProvider"]
        )

    def run(self, patches: np.ndarray, pos_embed: np.ndarray, scale_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched model call on tokenizer outputs. Returns (prediction [B, classes], score [B]).

        The model has no attention mask, so a batch must hold sequences of equal length;
        padding a shorter sequence changes its score.
        """
        feeds = dict(zip(INPUT_NAMES, (patches, pos_embed, scale_ids)))
        prediction, score = self.session.run(None, feeds)
        return prediction, score

    def score(self, image_path: str | Path) -> float:
        patches, pos_embed, scale_ids = self.tokenizer(image_path)
        _, score = self.run(patches[None], pos_embed[None], scale_ids[None])
        return float(score[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--dataset", choices=list(DATASETS), default="aadb")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    scorer = CharmScorer(args.models_dir / args.dataset, args.dataset)
    for image in args.images:
        print(f"{scorer.score(image):.4f}\t{image}")


if __name__ == "__main__":
    main()
