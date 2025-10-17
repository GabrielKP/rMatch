"""This script runs specified mutual information method to compute
the recall matrix for the first 10 cyoa participants."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from recall_matrix import console
from recall_matrix.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
)
from recall_matrix.mutual_information_recall_matrix import MIRM


def plot_recall_matrix_comparison(
    story_name: str,
    sub_id: str,
    recall_matrix_human_binary: np.ndarray,
    recall_matrix_mutual_information: np.ndarray,
    mutual_information_method: str,
    mutual_information_normalize: bool,
    model_name: str,
):
    """Plot the recall matrices comparison"""

    norm_str = "normalized" if mutual_information_normalize else "unnormalized"
    output_dir = (
        Path("outputs") / "test" / mutual_information_method / norm_str / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    m, n = recall_matrix_human_binary.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(int((n / m + 1) * 9), 9))

    # color_matrix = np.ones_like(rm)
    ax1.imshow(recall_matrix_human_binary, cmap="Reds")
    ax1.set_title("Human binary recall matrix")
    ax1.set_xlabel("Recall segments")
    ax1.set_ylabel("Story segments")
    ax1.set_aspect(1)

    im2 = ax2.imshow(recall_matrix_mutual_information, cmap="Reds")
    ax2.set_title("Mutual information recall matrix")
    ax2.set_xlabel("Recall segments")
    ax2.set_ylabel("Story segments")
    ax2.set_aspect(1)

    fig.colorbar(im2, ax=[ax1, ax2], label="Mutual information (bits)")
    fig.suptitle(
        (
            f"{story_name} | {sub_id} | {mutual_information_method} | {norm_str}"
            f" | {model_name}"
        )
    )

    fig.savefig(output_dir / f"{story_name}_{sub_id}.png", bbox_inches="tight")
    console.print(f"Saved plot to {output_dir / f'{story_name}_{sub_id}.png'}")
    plt.close(fig)


def test_mutual_information_method(
    mutual_information_method: str,
    mutual_information_normalize: bool,
    model_name: str | None = None,
    verbose: bool = False,
):
    # 1. load story & recall segments
    story_names = ["alice_2", "alice_3", "monthiversary_3", "monthiversary_4"]
    cyao_story_recall_segments = load_cyoa_story_recall_segments(
        story_names=story_names
    )

    # 2. init mirm
    mirm = MIRM(
        model_name=model_name,
        mutual_information_method=mutual_information_method,
        mutual_information_normalize=mutual_information_normalize,
    )
    model_name = mirm.model_name

    # 3. compute recall matrix and plot
    for (
        story_name,
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(cyao_story_recall_segments, desc="(story/sub_ids)", disable=verbose):
        # a) load human binary recall matrix
        rm_human_binary = load_cyoa_recall_matrix_human_binary(
            story_name=story_name, sub_id=sub_id
        )
        # b) compute mutual information recall matrix
        rm_mutual_information = mirm.compute_mutual_information_recall_matrix(
            story_segments=story_segments,
            recall_segments=recall_segments,
            verbose=verbose,
        )

        # c) plot both
        plot_recall_matrix_comparison(
            story_name=story_name,
            sub_id=sub_id,
            recall_matrix_human_binary=rm_human_binary,
            recall_matrix_mutual_information=rm_mutual_information,
            mutual_information_method=mutual_information_method,
            mutual_information_normalize=mutual_information_normalize,
            model_name=model_name,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-mm", "--mutual_information_method", type=str, required=True)
    parser.add_argument("-mn", "--mutual_information_normalize", action="store_true")
    parser.add_argument("-vm", "--verbose", action="store_true")
    args = parser.parse_args()

    test_mutual_information_method(
        mutual_information_method=args.mutual_information_method,
        mutual_information_normalize=args.mutual_information_normalize,
        verbose=args.verbose,
    )
