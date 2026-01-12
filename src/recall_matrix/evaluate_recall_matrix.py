"""This script runs specified mutual information method to compute
the recall matrix for the first 10 cyoa participants."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from recall_matrix import console
from recall_matrix.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
    load_nfrd_story_recall_segments,
)
from recall_matrix.recall_matrix.mutual_information import MIRM
from recall_matrix.recall_matrix.reranker import RRRM
from recall_matrix.recall_matrix.reranker_2 import RRRM2


def evaluate_recall_matrix_method(
    method: str,
    mi_normalize: bool = False,
    random: bool = False,
    random_rows: bool = False,
    binary: bool = False,
    model_name: str | None = None,
    verbose: bool = False,
    debug: bool = False,
    seed: int = 123,
):
    if debug:
        verbose = True

    assert not (random and random_rows), "Cannot use both random and random_rows"

    # 1. load story & recall segments
    story_recall_segments = load_cyoa_story_recall_segments()

    # 2. init recall matrix object
    if not (random or random_rows):
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
        elif method.startswith("rerank2"):
            reranker_method = method[len("rerank2_") :]
            rmo = RRRM2(
                model_name=model_name,
                reranker_method=reranker_method,
                reranker_binary=binary,
                debug=debug,
            )
        elif method.startswith("rerank"):
            reranker_method = method[len("rerank_") :]
            rmo = RRRM(
                model_name=model_name,
                reranker_method=reranker_method,
                reranker_binary=binary,
                debug=debug,
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        model_name = rmo.model_name
    else:
        rmo = None

    rng = np.random.default_rng(seed)

    # 3. compute recall matrices and evaluate
    precisions = list()
    recalls = list()
    f1s = list()
    rm_models = list()
    rm_controls = list()
    for (
        story_name,
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(story_recall_segments, desc="(eval)", disable=verbose):
        # a) compute recall matrices
        rm_control = load_cyoa_recall_matrix_human_binary(
            story_name=story_name, sub_id=sub_id
        )
        if random:
            flat = rng.permutation(rm_control.flatten())
            rm_model = flat.reshape(rm_control.shape)
        elif random_rows:
            rm_model = rng.permutation(rm_control)
        else:
            assert rmo is not None, "rmo must be initialized"
            rm_model = rmo.compute_recall_matrix(
                story_segments=story_segments,
                recall_segments=recall_segments,
                verbose=verbose,
            )

        # b) evaluate
        rm_model_flat = rm_model.flatten()
        rm_control_flat = rm_control.flatten()
        rm_models.append(rm_model_flat)
        rm_controls.append(rm_control_flat)

        precision = precision_score(rm_control_flat, rm_model_flat)
        recall = recall_score(rm_control_flat, rm_model_flat)
        f1 = f1_score(rm_control_flat, rm_model_flat)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # https://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html
    # Micro:
    rm_models_flat = np.concatenate(rm_models)
    rm_controls_flat = np.concatenate(rm_controls)
    precision_micro = precision_score(rm_controls_flat, rm_models_flat)
    recall_micro = recall_score(rm_controls_flat, rm_models_flat)
    f1_micro = f1_score(rm_controls_flat, rm_models_flat)
    micro_str = (
        f"Micro: F1={f1_micro:.3f},"
        f" Precision={precision_micro:.3f},"
        f" Recall={recall_micro:.3f}"
    )

    # Macro:
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = (2 * precision_macro * recall_macro) / (precision_macro + recall_macro)
    macro_str = (
        f"Macro: F1={f1_macro:.3f},"
        f" Precision={precision_macro:.3f},"
        f" Recall={recall_macro:.3f}"
    )
    if random:
        method_str = "random"
    elif random_rows:
        method_str = "random_rows"
    else:
        method_str = method
    console.print(f"Method: {method_str} | N recalls: {len(story_recall_segments)}")
    console.print(micro_str)
    console.print(macro_str)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--method", type=str, required=True)
    parser.add_argument("-mn", "--mi_normalize", action="store_true")
    parser.add_argument("-r", "--random", action="store_true")
    parser.add_argument("-rr", "--random_rows", action="store_true")
    parser.add_argument("-M", "--model_name", type=str, default=None)
    parser.add_argument("-vm", "--verbose", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    evaluate_recall_matrix_method(
        method=args.method,
        mi_normalize=args.mi_normalize,
        random=args.random,
        random_rows=args.random_rows,
        binary=True,
        model_name=args.model_name,
        verbose=args.verbose,
        debug=args.debug,
    )
