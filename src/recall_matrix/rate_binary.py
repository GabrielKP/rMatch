import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from recall_matrix import print_config
from recall_matrix.load import load_story_recall_segments
from recall_matrix.recall_matrix.reranker import RRRM


def rate_binary(
    story_name: str,
    model_name: str,
    reranker_threshold: float,
    story_segment_method: str | None = None,
    recall_segment_method: str | None = None,
    suffix: str | None = None,
    output_scores: bool = False,
    top_k: int = 5,
    device: str | None = None,
):
    story_recall_segments, story_segment_method, recall_segment_method = (
        load_story_recall_segments(
            story_name=story_name,
            story_segment_method=story_segment_method,
            recall_segment_method=recall_segment_method,
        )
    )

    suffix_str = f"-{suffix}" if suffix else ""
    recall_segment_method_str = f"-rsm_{recall_segment_method}"
    story_segment_method_str = f"-ssm_{story_segment_method}"
    top_k_str = f"-tk_{top_k}"
    output_scores_str = "-os" if output_scores else ""
    param_str = (
        f"reranker_thresholded_{reranker_threshold}"
        f"{recall_segment_method_str}"
        f"{story_segment_method_str}"
        f"{top_k_str}"
        f"{output_scores_str}"
        f"{suffix_str}"
    )

    output_dict = {
        "param_str": param_str,
        "story_name": story_name,
        "story_segment_method": story_segment_method,
        "recall_segment_method": recall_segment_method,
        "model_name": model_name,
        "reranker_threshold": reranker_threshold,
        "top_k": top_k,
        "output_scores": output_scores,
    }
    print_config(output_dict)

    reranker_rmo = RRRM(
        model_name=model_name,
        reranker_method="thresholded",
        reranker_threshold=reranker_threshold,
        reranker_binary=True,
        device=device,
        debug=False,
        top_k=top_k,
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
            if output_scores:
                topk_story_segment_indices = np.where(rm_reranker[:, idx_recall])[0]
                topk_story_segment_scores = rm_reranker[
                    topk_story_segment_indices, idx_recall
                ]
                story_segment_indices.append(
                    (
                        idx_recall,
                        [
                            (idx, score)
                            for idx, score in zip(
                                topk_story_segment_indices, topk_story_segment_scores
                            )
                        ],
                    )
                )
            else:
                story_segment_indices.append(
                    (idx_recall, np.where(rm_reranker[:, idx_recall])[0].tolist())
                )
        story_segment_indices_dict[sub_id] = story_segment_indices

    output_dict["ratings"] = story_segment_indices_dict

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
    args.add_argument(
        "-rt",
        "--reranker_threshold",
        type=float,
        default=0.09,
        help="threshold above which a story-segment score counts as recalled.",
    )
    args.add_argument(
        "--output_scores",
        action="store_true",
        default=False,
        help=(
            "Outputs the scores of the story segments for each recall segment."
            " The output dict then contains for each recall segment a dict with"
            " story_segment_index -> score."
        ),
    )
    args.add_argument(
        "-tk",
        "--top_k",
        type=int,
        default=5,
        help="Picks top k candidate story segments for each recall segment.",
    )
    args.add_argument(
        "--suffix", type=str, default=None, help="Suffix for the output file"
    )
    args.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for the reranker. If None, will be autoselected.",
    )
    args = args.parse_args()
    rate_binary(
        story_name=args.story_name,
        story_segment_method=args.story_segment_method,
        recall_segment_method=args.recall_segment_method,
        model_name=args.model_name,
        reranker_threshold=args.reranker_threshold,
        top_k=args.top_k,
        output_scores=args.output_scores,
        suffix=args.suffix,
        device=args.device,
    )
