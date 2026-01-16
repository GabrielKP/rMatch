import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from recall_matrix.load import load_story_recall_segments
from recall_matrix.recall_matrix.reranker import RRRM


def rate_binary(
    story_name: str,
    model_name: str,
    reranker_threshold: float,
    story_segment_method: str | None = None,
    recall_segment_method: str | None = None,
    suffix: str | None = None,
):
    story_recall_segments, story_segment_method, recall_segment_method = (
        load_story_recall_segments(
            story_name=story_name,
            story_segment_method=story_segment_method,
            recall_segment_method=recall_segment_method,
        )
    )

    reranker_rmo = RRRM(
        model_name=model_name,
        reranker_method="thresholded",
        reranker_threshold=reranker_threshold,
        reranker_binary=True,
        debug=False,
    )

    story_segment_indices_dict: dict[str, list[int]] = dict()
    for (
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(story_recall_segments, desc="(story/sub_ids)"):
        # a) compute reranker recall matrix
        rm_reranker = reranker_rmo.compute_recall_matrix(
            story_segments=story_segments,
            recall_segments=recall_segments,
        )
        # convert to indices (preserving order)
        n_recalls = rm_reranker.shape[1]
        story_segment_indices = []
        for idx_recall in range(n_recalls):
            story_segment_indices.append(
                {idx_recall: np.where(rm_reranker[:, idx_recall])[0].tolist()}
            )
        story_segment_indices_dict[sub_id] = story_segment_indices

    output_dict = {
        "story_name": story_name,
        "story_segment_method": story_segment_method,
        "recall_segment_method": recall_segment_method,
        "model_name": model_name,
        "reranker_threshold": reranker_threshold,
        "ratings": story_segment_indices_dict,
    }
    suffix_str = f"-{suffix}" if suffix else ""
    recall_segment_method_str = f"-rsm_{recall_segment_method}"
    story_segment_method_str = f"-ssm_{story_segment_method}"
    param_str = (
        f"reranker_thresholded_{reranker_threshold}"
        f"{recall_segment_method_str}"
        f"{story_segment_method_str}"
        f"{suffix_str}"
    )
    output_path = (
        Path("data")
        / "stories-and-recalls"
        / story_name
        / "ratings"
        / f"{param_str}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        f_out.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-s", "--story_name", type=str, default="pieman")
    args.add_argument("-ssm", "--story_segment_method", type=str, default=None)
    args.add_argument("-rsm", "--recall_segment_method", type=str, default=None)
    args.add_argument("-m", "--model_name", type=str, default="BAAI/bge-reranker-v2-m3")
    args.add_argument("-rt", "--reranker_threshold", type=float, default=0.09)
    args.add_argument(
        "--suffix", type=str, default=None, help="Suffix for the output file"
    )
    args = args.parse_args()
    rate_binary(
        story_name=args.story_name,
        story_segment_method=args.story_segment_method,
        recall_segment_method=args.recall_segment_method,
        model_name=args.model_name,
        reranker_threshold=args.reranker_threshold,
        suffix=args.suffix,
    )
