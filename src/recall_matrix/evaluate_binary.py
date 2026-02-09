import argparse
import datetime
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from recall_matrix import console
from recall_matrix.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
    load_ratings_dict,
    load_story_recall_segments,
)
from recall_matrix.raters import initialize_rater
from recall_matrix.utils import ratings_single_sub_to_matrix


def eval_param_str(
    rater_name: str,
    model_name: str,
    seed: int,
    random_mode: str | None,
) -> str:
    """Get the param string for the evaluation."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_mode_str = f"-random_mode_{random_mode}" if random_mode is not None else ""
    model_name_str = model_name.replace("/", "_")
    param_str = (
        f"{timestamp}-{rater_name}-m_{model_name_str}-seed_{seed}{random_mode_str}"
    )
    return param_str


def accuracy(array_1: np.ndarray, array_2: np.ndarray) -> float:
    return np.sum(array_1 == array_2) / len(array_1)


def evaluate(
    rater_name: str,
    model_name: str,
    testset: str,
    device: str | None = None,
    seed: int = 42,
    random_mode: str | None = None,
    reranker_threshold: float | None = None,
    top_k: int = 5,
):
    """Evaluate the rater."""
    rater = initialize_rater(
        rater_name=rater_name, model_name=model_name, device=device
    )

    output_dir = (
        Path("data")
        / "eval"
        / eval_param_str(rater_name, model_name, seed, random_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # load story recall segments
    if testset == "cyoa":
        story_recall_segments = load_cyoa_story_recall_segments()
    elif testset == "memsearch":
        story_names_memsearch = [
            "breadland",
            "ednora",
            "from_dad_to_son",
            "heartstrings",
            "hollow",
            "i_love_death",
            "ichthys",
            "laundry",
            "mismatched",
            "mop",
            "my_cat_lucy",
            "numb",
            "queen_of_basketball",
            "stapler",
            "synesthesia",
            "the_docks",
            "the_gift",
            "the_port",
            "the_soup",
            "thief",
        ]
        story_recall_segments_memsearch = list()
        for story_name in story_names_memsearch:
            story_recall_segments_single, _, _ = load_story_recall_segments(
                story_name=story_name,
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            story_recall_segments_memsearch.extend(
                [
                    (story_name, sub_id, story_segs, recall_segs)
                    for sub_id, story_segs, recall_segs in story_recall_segments_single
                ]
            )

        story_recall_segments = story_recall_segments_memsearch

        # pre load rating dicts for memsearch
        ratings_dicts_memsearch = defaultdict(dict)
        for story_name in story_names_memsearch:
            ratings_dict = load_ratings_dict(
                story_name=story_name,
                rater_name="human",
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            n_story_segments = ratings_dict["n_story_segments"]
            # convert to matrix
            for sub_id, single_sub_ratings in ratings_dict["ratings"].items():
                rm_memsearch = ratings_single_sub_to_matrix(
                    single_sub_ratings, n_story_segments
                )
                ratings_dicts_memsearch[story_name][sub_id] = rm_memsearch
    else:
        raise ValueError(f"Invalid testset: {testset}")

    rng = np.random.default_rng(seed)

    precisions = list()
    recalls = list()
    pearsonrs = list()
    accuracies = list()
    recall_matrices_model: list[np.ndarray] = list()
    recall_matrices_comparison: list[np.ndarray] = list()
    for (
        story_name,
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(story_recall_segments, desc="(eval)"):
        # a) get ground truth
        if testset == "cyoa":
            rm_comparison = load_cyoa_recall_matrix_human_binary(
                story_name=story_name, sub_id=sub_id
            )
        else:
            rm_comparison = ratings_dicts_memsearch[story_name][sub_id]  # type: ignore

        # b) get model ratings
        if random_mode == "full_shuffle":
            flat = rng.permutation(rm_comparison.flatten())
            rm_model = flat.reshape(rm_comparison.shape)
        elif random_mode == "row_shuffle":
            rm_model = rng.permutation(rm_comparison)
        else:
            if rater_name == "reranker":
                single_sub_ratings = rater.compute_ratings_single_sub(
                    story_segments=story_segments,
                    recall_segments=recall_segments,
                    output_scores=False,
                    threshold=reranker_threshold,  # type: ignore
                    top_k=top_k,  # type: ignore
                )
            else:
                single_sub_ratings = rater.compute_ratings_single_sub(
                    story_segments=story_segments,
                    recall_segments=recall_segments,
                    output_scores=False,
                )
            rm_model = ratings_single_sub_to_matrix(
                single_sub_ratings,  # type: ignore
                len(story_segments),
            )

        # b) evaluate
        recall_matrices_model.append(rm_model)
        recall_matrices_comparison.append(rm_comparison)

        if (rm_model == 0).all():
            precision = 0
            recall = 0
            pearsonr_score = 0
            accuracy_score = 0
        else:
            rm_model_flat = rm_model.flatten()
            rm_comparison_flat = rm_comparison.flatten()
            precision = precision_score(rm_comparison_flat, rm_model_flat)
            recall = recall_score(rm_comparison_flat, rm_model_flat)
            pearsonr_score = pearsonr(rm_comparison_flat, rm_model_flat)[0]  # type: ignore
            accuracy_score = accuracy(rm_comparison_flat, rm_model_flat)

        precisions.append(precision)
        recalls.append(recall)
        pearsonrs.append(pearsonr_score)
        accuracies.append(accuracy_score)

    # output results
    if random_mode is not None:
        rater_str = random_mode
    else:
        rater_str = rater_name
    console.print(f"Rater: {rater_str} | N recalls: {len(story_recall_segments)}")

    # Macro F1 (treat each recall matrix independently)
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = (2 * precision_macro * recall_macro) / (precision_macro + recall_macro)
    macro_f1_str = (
        f"Macro: F1={f1_macro:.3f},"
        f" Precision={precision_macro:.3f},"
        f" Recall={recall_macro:.3f}"
    )

    console.print(macro_f1_str)

    accuracy_macro = np.mean(accuracies)
    console.print(f"Accuracy: {accuracy_macro:.3f}")

    pearsonr_macro = np.mean(pearsonrs)
    console.print(f"Pearsonr: {pearsonr_macro:.3f}")

    results_dict = {
        "rater_name": rater_name,
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "random_mode": random_mode,
        "macro_f1": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "accuracy_macro": accuracy_macro,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f)
    console.print(f"Saved results to {results_path}")

    # save recall matrices
    recall_matrices_model_path = output_dir / "recall_matrices_model.pkl"
    recall_matrices_comparison_path = output_dir / f"recall_matrices_{testset}.pkl"

    with open(recall_matrices_model_path, "wb") as f:
        pickle.dump(recall_matrices_model, f)
    with open(recall_matrices_comparison_path, "wb") as f:
        pickle.dump(recall_matrices_comparison_path, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-r",
        "--rater_name",
        choices=["reranker", "openai", "huggingface"],
        default="reranker",
        help="Name of the rater to use. Default is 'reranker'.",
    )
    args.add_argument(
        "-t",
        "--testset",
        choices=["memsearch", "cyoa"],
        default="cyoa",
        help="Name of the testset to use. Default is 'cyoa'.",
    )
    args.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help=(
            "[reranker, openai, huggingface] Name of the model to use for the reranker."
        ),
    )
    args.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "[reranker, huggingface] Device to use for the reranker."
            "If None, will be autoselected."
        ),
    )
    args.add_argument(
        "--random_mode",
        choices=["full_shuffle", "row_shuffle", None],
        default=None,
        help="Mode for the random shuffle. If None, will not shuffle.",
    )
    args.add_argument(
        "-rt",
        "--reranker_threshold",
        type=float,
        default=0.09,
        help=(
            "[reranker] Threshold above which a story-segment score counts as recalled."
        ),
    )
    args.add_argument(
        "-tk",
        "--top_k",
        type=int,
        default=5,
        help=(
            "[reranker] Number of top k story segments to consider"
            " for each recall segment."
        ),
    )
    args = args.parse_args()
    evaluate(
        rater_name=args.rater_name,
        testset=args.testset,
        model_name=args.model_name,
        device=args.device,
        random_mode=args.random_mode,
        reranker_threshold=args.reranker_threshold,
        top_k=args.top_k,
    )
