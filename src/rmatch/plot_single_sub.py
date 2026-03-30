"""Plot a single subject's recall matrix for multiple raters.

Example usage:
uv run src/rmatch/plot_single_sub.py\
    sub-001\
    data/stories-and-recalls/pieman/ratings/lda_hmm.json
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmatch.utils import ratings_single_sub_to_matrix


def plot_single_sub_recall_matrices(
    story_name: str,
    sub_id: str,
    story_segmentation_method: str,
    recall_segmentation_method: str,
    recall_matrices: list[np.ndarray],
    rater_names: list[str],
):
    """Plot the recall matrices comparison"""

    assert len(recall_matrices) > 0

    output_dir = (
        Path("data") / "stories-and-recalls" / story_name / "plots" / "recall_matrices"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_recall_matrices = len(recall_matrices)

    n_story_segments, n_recall_segments = recall_matrices[0].shape

    width = int((n_recall_segments / n_story_segments + 1) * 9)
    height = 9
    fig, axes = plt.subplots(
        1, n_recall_matrices, figsize=(width * n_recall_matrices, height)
    )

    if n_recall_matrices == 1:
        axes = [axes]

    for idx, (rater_name, recall_matrix) in enumerate(
        zip(rater_names, recall_matrices)
    ):
        ax = axes[idx]
        img = ax.imshow(recall_matrix, cmap="Reds")
        ax.set_title(rater_name)
        ax.set_xlabel("Recall segments")
        ax.set_ylabel("Story segments")
        ax.set_aspect(1)

    fig.colorbar(img, ax=axes, label="Recalled")  # type: ignore
    fig.suptitle(
        (
            f"{story_name}"
            f" | {sub_id}"
            f" | ssm: {story_segmentation_method}"
            f" | rsm: {recall_segmentation_method}"
        )
    )

    output_path = (
        output_dir
        / f"{sub_id}-ssm_{story_segmentation_method}-rsm_{recall_segmentation_method}.png"  # noqa: E501
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_single_sub(sub_id: str, paths_ratings: list[Path]):
    """Plot a single subject's recall matrix"""
    # TODO: plot_single_sub cannot handle output_scores=True

    assert len(paths_ratings) > 0, "need at least one rating file"

    recall_matrices = list()
    rater_names = list()
    for paths_rating in paths_ratings:
        data_dict = json.loads(paths_rating.read_text())

        recall_matrix = ratings_single_sub_to_matrix(
            data_dict["ratings"][sub_id], data_dict["n_story_segments"]
        )

        recall_matrices.append(recall_matrix)
        rater_names.append(data_dict["rater_name"])
        story_segmentation_method = data_dict["story_segmentation_method"]
        recall_segmentation_method = data_dict["recall_segmentation_method"]
        story_name = data_dict["story_name"]

    plot_single_sub_recall_matrices(
        story_name=story_name,  # type: ignore
        sub_id=sub_id,
        story_segmentation_method=story_segmentation_method,  # type: ignore
        recall_segmentation_method=recall_segmentation_method,  # type: ignore
        recall_matrices=recall_matrices,
        rater_names=rater_names,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "sub_id",
        type=str,
    )
    parser.add_argument(
        "paths_ratings",
        nargs="+",
        type=Path,
    )
    args = parser.parse_args()
    plot_single_sub(sub_id=args.sub_id, paths_ratings=args.paths_ratings)
