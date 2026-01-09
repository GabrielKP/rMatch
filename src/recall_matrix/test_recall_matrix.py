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
    load_nfrd_recall_matrix_human_mi,
    load_nfrd_story_recall_segments,
)
from recall_matrix.mutual_information_recall_matrix import MIRM
from recall_matrix.reranker_recall_matrix import RRRM
from recall_matrix.reranker_recall_matrix_2 import RRRM2


def plot_recall_matrix_comparison(
    story_name: str,
    sub_id: str,
    recall_matrix_control: np.ndarray,
    recall_matrix_method: np.ndarray,
    method: str,
    method_postfix: str,
    variant_str: str,
    measure_label: str,
    model_name: str,
    control_title: str,
):
    """Plot the recall matrices comparison"""

    method_str = f"{method}{method_postfix}"

    output_dir = Path("outputs") / "test" / method_str / variant_str / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    m, n = recall_matrix_control.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(int((n / m + 1) * 9), 9))

    # color_matrix = np.ones_like(rm)
    ax1.imshow(recall_matrix_control, cmap="Reds")
    ax1.set_title(control_title)
    ax1.set_xlabel("Recall segments")
    ax1.set_ylabel("Story segments")
    ax1.set_aspect(1)

    im2 = ax2.imshow(recall_matrix_method, cmap="Reds")
    ax2.set_title(f"{method_str} recall matrix")
    ax2.set_xlabel("Recall segments")
    ax2.set_ylabel("Story segments")
    ax2.set_aspect(1)

    fig.colorbar(im2, ax=[ax1, ax2], label=measure_label)
    fig.suptitle(
        (f"{story_name} | {sub_id} | {method_str} | {variant_str} | {model_name}")
    )

    fig.savefig(output_dir / f"{story_name}_{sub_id}.png", bbox_inches="tight")
    console.print(f"Saved plot to {output_dir / f'{story_name}_{sub_id}.png'}")
    plt.close(fig)


def test_recall_matrix_method(
    method: str,
    mi_normalize: bool = False,
    binary: bool = False,
    model_name: str | None = None,
    pieman: bool = False,
    verbose: bool = False,
    debug: bool = False,
):
    if debug:
        verbose = True

    # 1. load story & recall segments
    if pieman:
        story_recall_segments = load_nfrd_story_recall_segments(
            story_names=["pieman"],
            story_segment_method="sentence",
            sub_ids=["P1"],
        )
    else:
        story_names = ["alice_2", "alice_3", "monthiversary_3", "monthiversary_4"]
        story_recall_segments = load_cyoa_story_recall_segments(story_names=story_names)

    # 2. init recall matrix object
    if method.startswith("mi"):
        y_instruction = (
            "Your task is to predict what a human would recall about a"
            " story segment."
            "\nThe following is the story segment:\n"
        )
        x_instruction = "\nThe following is the matching recall segment:\n"
        x_instruction_no_y = "\nThe following is a human recall segment:\n"

        mi_method = method[len("mi_") :]
        rmo = MIRM(
            model_name=model_name,
            mutual_information_method=mi_method,
            mutual_information_normalize=mi_normalize,
            y_instruction=y_instruction,
            x_instruction=x_instruction,
            x_instruction_no_y=x_instruction_no_y,
            debug=debug,
        )
        variant_str = "normalized" if mi_normalize else "unnormalized"
        measure_label = (
            "Mutual information" if mi_normalize else "Mutual information (bits)"
        )
        method_postfix = ""
    elif method.startswith("rerank2"):
        reranker_method = method[len("rerank2_") :]
        rmo = RRRM2(
            model_name=model_name,
            reranker_method=reranker_method,
            reranker_binary=binary,
            debug=debug,
        )
        variant_str = "binary" if binary else "score"
        measure_label = "Recalled" if binary else "Score"
        method_postfix = ""
    elif method.startswith("rerank"):
        reranker_method = method[len("rerank_") :]
        rmo = RRRM(
            model_name=model_name,
            reranker_method=reranker_method,
            reranker_binary=binary,
            debug=debug,
        )
        variant_str = "binary" if binary else "score"
        measure_label = "Recalled" if binary else "Score"
        method_postfix = f"_{str(rmo.reranker_threshold)}"
    else:
        raise ValueError(f"Invalid method: {method}")

    model_name = rmo.model_name

    # 3. compute recall matrix and plot
    for (
        story_name,
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(story_recall_segments, desc="(story/sub_ids)", disable=verbose):
        # a) compute mutual information recall matrix
        rm_mutual_information = rmo.compute_recall_matrix(
            story_segments=story_segments,
            recall_segments=recall_segments,
            verbose=verbose,
        )

        # b) load control recall matrix
        if pieman:
            rm_control = load_nfrd_recall_matrix_human_mi(
                story_name=story_name, sub_id=sub_id, rater="dhruva"
            )
            control_title = "Human MI recall matrix"
        else:
            rm_control = load_cyoa_recall_matrix_human_binary(
                story_name=story_name, sub_id=sub_id
            )
            control_title = "Human binary recall matrix"

        # c) plot both
        plot_recall_matrix_comparison(
            story_name=story_name,
            sub_id=sub_id,
            recall_matrix_control=rm_control,
            recall_matrix_method=rm_mutual_information,
            method=method,
            method_postfix=method_postfix,
            variant_str=variant_str,
            measure_label=measure_label,
            model_name=model_name,
            control_title=control_title,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--method", type=str, required=True)
    parser.add_argument("-mn", "--mi_normalize", action="store_true")
    parser.add_argument("-b", "--binary", action="store_true")
    parser.add_argument("-M", "--model_name", type=str, default=None)
    parser.add_argument("-p", "--pieman", action="store_true")
    parser.add_argument("-vm", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    test_recall_matrix_method(
        method=args.method,
        mi_normalize=args.mi_normalize,
        binary=args.binary,
        model_name=args.model_name,
        pieman=args.pieman,
        verbose=args.verbose,
        debug=args.debug,
    )
