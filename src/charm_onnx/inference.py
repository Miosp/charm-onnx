"""Score images with an exported Charm model using only ONNX Runtime and NumPy."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from charm_onnx.specs import DATASETS, HF_ONNX_REPO_ID, INPUT_NAMES, MODEL_FILE, POS_EMBED_FILE
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

    @classmethod
    def from_pretrained(cls, dataset: str, repo_id: str = HF_ONNX_REPO_ID, **kwargs) -> "CharmScorer":
        """Download (or reuse cached) exported files for `dataset` from the Hugging Face Hub."""
        for filename in (MODEL_FILE, POS_EMBED_FILE):
            path = hf_hub_download(repo_id, f"{dataset}/{filename}")
        return cls(Path(path).parent, dataset, **kwargs)

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
    parser.add_argument("--models-dir", type=Path, help="local export dir (default: download from the Hub)")
    args = parser.parse_args()
    if args.models_dir is None:
        scorer = CharmScorer.from_pretrained(args.dataset)
    else:
        scorer = CharmScorer(args.models_dir / args.dataset, args.dataset)
    for image in args.images:
        print(f"{scorer.score(image):.4f}\t{image}")


if __name__ == "__main__":
    main()
