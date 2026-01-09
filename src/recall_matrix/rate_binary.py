import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from recall_matrix.load import load_nfrd_story_recall_segments
from recall_matrix.reranker_recall_matrix import RRRM


def rate_binary():
    story_name = "pieman"
    story_segment_method = "behavioral"
    model_name = "BAAI/bge-reranker-v2-m3"
    debug = False
    verbose = True

    story_recall_segments = load_nfrd_story_recall_segments(
        story_names=[story_name],
        story_segment_method=story_segment_method,
        sub_ids=[
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "P10",
            "P11",
            "P12",
        ],
    )

    reranker_rmo = RRRM(
        model_name=model_name,
        reranker_method="thresholded",
        reranker_binary=True,
        debug=debug,
    )

    story_segment_indices_dict: dict[str, list[int]] = dict()
    for (
        story_name,
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(
        story_recall_segments,
        desc="(story/sub_ids)",
        disable=verbose,
    ):
        # a) compute reranker recall matrix
        rm_reranker = reranker_rmo.compute_recall_matrix(
            story_segments=story_segments,
            recall_segments=recall_segments,
            verbose=verbose,
        )
        # convert indices
        story_segment_indices = np.where(rm_reranker.any(axis=1))[0]
        story_segment_indices_dict[sub_id] = story_segment_indices.tolist()

    output_path = (
        Path("outputs")
        / "binary"
        / f"{story_name}_{story_segment_method}_indices_dict.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f_out:
        json.dump(story_segment_indices_dict, f_out)


if __name__ == "__main__":
    rate_binary()
