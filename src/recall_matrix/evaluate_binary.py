import argparse
import datetime
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from recall_matrix import console
from recall_matrix.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
    load_ratings_dict,
    load_story_recall_segments,
)
from recall_matrix.raters import initialize_rater
from recall_matrix.raters.rater import Rater
from recall_matrix.utils import ratings_single_sub_to_matrix


def eval_param_str(
    repeat_reliability: bool,
    testset: str,
    rater_name: str,
    model_name: str | None,
    seed: int,
    random_mode: str | None,
) -> str:
    """Get the param string for the evaluation."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rr_str = "-rr" if repeat_reliability else ""
    random_mode_str = f"-random_mode_{random_mode}" if random_mode is not None else ""

    model_name_str = ""
    if model_name is not None:
        model_name_str = f"-m_{model_name.replace('/', '_')}"
    param_str = (
        f"{timestamp}{rr_str}-{testset}"
        f"-{rater_name}{model_name_str}-seed_{seed}{random_mode_str}"
    )
    return param_str


def accuracy(array_1: np.ndarray, array_2: np.ndarray) -> float:
    if len(array_1) == 0:
        return 0.0
    return np.sum(array_1 == array_2) / len(array_1)


def load_story_recall_segments_default(
    testset: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, dict[str, np.ndarray]] | None,
]:
    """Returns story, recall pairs and human annotations."""

    human_ratings_dict = defaultdict(dict)
    if testset.startswith("cyoa"):
        if testset == "cyoa_alice10":
            story_names = [
                "alice_2",
                "alice_3",
                "alice_4",
                "alice_5",
                "alice_6",
                "alice_7",
                "alice_8",
                "alice_11",
                "alice_12",
                "alice_13",
            ]
        elif testset == "cyoa_monthiversary6":
            story_names = [
                "monthiversary_3",
                "monthiversary_4",
                "monthiversary_10",
                "monthiversary_14",
                "monthiversary_19",
                "monthiversary_23",
                "monthiversary_25",
            ]
        else:
            story_names = None
        story_recall_segments = load_cyoa_story_recall_segments(story_names=story_names)

        # load human annotations
        for story_name, sub_id, _, _ in story_recall_segments:
            human_ratings_dict[story_name][sub_id] = (
                load_cyoa_recall_matrix_human_binary(
                    story_name=story_name, sub_id=sub_id
                )
            )

    elif testset.startswith("memsearch"):
        if testset == "memsearch10":
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
            ]
        else:
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
                human_ratings_dict[story_name][sub_id] = rm_memsearch
    else:
        raise ValueError(f"Invalid testset: {testset}")
    return story_recall_segments, human_ratings_dict


def get_model_ratings(
    random_mode: str | None,
    rng: np.random.Generator,
    rm_comparison: np.ndarray,
    rater: Rater,
    story_segments: list[str],
    recall_segments: list[str],
):
    if random_mode == "full_shuffle":
        flat = rng.permutation(rm_comparison.flatten())
        rm_model = flat.reshape(rm_comparison.shape)
    elif random_mode == "row_shuffle":
        rm_model = rng.permutation(rm_comparison)
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
    return rm_model


def evaluate(
    repeat_reliability: bool,
    rater_name: str,
    model_name: str | None,
    testset: str,
    device: str | None = None,
    seed: int = 42,
    random_mode: str | None = None,
    window_size: int = 5,
    dry_run: bool = False,
    movie_mode: bool = False,
    reranker_threshold: float | None = None,
    top_k: int = 5,
):
    """Evaluate the rater."""
    rater = initialize_rater(
        rater_name=rater_name,
        model_name=model_name,
        device=device,
        reranker_threshold=reranker_threshold,
        window_size=window_size,
        dry_run=dry_run,
        top_k=top_k,
        movie_mode=movie_mode,
    )
    if hasattr(rater, "model_name"):
        model_name = rater.model_name  # type: ignore
    else:
        model_name = None

    output_dir = (
        Path("data")
        / "eval"
        / eval_param_str(
            repeat_reliability=repeat_reliability,
            testset=testset,
            rater_name=rater_name,
            model_name=model_name,
            seed=seed,
            random_mode=random_mode,
        )
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    story_recall_segments, human_ratings_dict = load_story_recall_segments_default(
        testset=testset
    )

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
        rm_comparison = human_ratings_dict[story_name][sub_id]  # type: ignore

        if (rm_comparison == 0).all():
            console.print(
                f"Skipping {story_name=} {sub_id=} : comparison matrix is all zero"
            )
            continue

        # b) get model ratings
        rm_model = get_model_ratings(
            random_mode=random_mode,
            rng=rng,
            rm_comparison=rm_comparison,
            rater=rater,
            story_segments=story_segments,
            recall_segments=recall_segments,
        )

        # b) evaluate
        recall_matrices_model.append(rm_model)
        recall_matrices_comparison.append(rm_comparison)

        if (rm_model == 0).all() or dry_run:
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

    if dry_run:
        console.print(f"[DRY RUN] Estimated Usage: {rater.get_usage()}")
        return

    # output results
    if random_mode is not None:
        rater_str = random_mode
    else:
        rater_str = rater_name
    console.print(
        f"Rater: {rater_str} | N recalls: {len(precisions)}"
        f" (evaluated) / {len(story_recall_segments)} (total)"
    )

    if not precisions:
        raise ValueError(
            "No recalls were evaluated (all comparison matrices were "
            "all-zero or empty)."
        )

    # Macro F1 (treat each recall matrix independently)
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    denom = precision_macro + recall_macro
    f1_macro = (2 * precision_macro * recall_macro) / denom if denom != 0 else 0.0
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
        "testset": testset,
        "rater_name": rater_name,
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "random_mode": random_mode,
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "accuracy_macro": float(accuracy_macro),
        "pearsonr_macro": float(pearsonr_macro),
    }

    if rater.get_usage() is not None:
        console.print(f"Total API usage: {rater.get_usage()}")
        results_dict["usage"] = rater.get_usage()

    output_dir.mkdir(parents=True, exist_ok=True)  # make again, in case user deleted it
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    console.print(f"Saved results to {results_path}")

    # save recall matrices
    recall_matrices_model_path = output_dir / "recall_matrices_model.pkl"
    recall_matrices_comparison_path = output_dir / f"recall_matrices_{testset}.pkl"

    with open(recall_matrices_model_path, "wb") as f:
        pickle.dump(recall_matrices_model, f)
    with open(recall_matrices_comparison_path, "wb") as f:
        pickle.dump(recall_matrices_comparison, f)


def load_story_recall_segments_repeat_reliability(
    testset: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, dict[str, np.ndarray]] | None,
]:
    # load story recall segments
    human_ratings_dict = defaultdict(dict)
    if testset.startswith("cyoa"):
        if testset == "cyoa_alice10":
            story_names = ["alice_3", "alice_6", "alice_12"]
        elif testset == "cyoa_monthiversary6":
            story_names = ["monthiversary_3", "monthiversary_4", "monthiversary_25"]
        else:
            raise ValueError(
                f"Invalid testset: {testset} -"
                " choose between 'cyoa_alice10' and 'cyoa_monthiversary6'"
            )
        all_story_recall_segments = load_cyoa_story_recall_segments(story_names=None)

        story_recall_segments = list()
        chosen_story_names = set()
        for (
            story_name,
            sub_id,
            story_segments,
            recall_segments,
        ) in all_story_recall_segments:
            if story_name in story_names and story_name not in chosen_story_names:
                story_recall_segments.append(
                    (story_name, sub_id, story_segments, recall_segments)
                )
                # only choose first recall per story
                chosen_story_names.add(story_name)
                if len(chosen_story_names) == len(story_names):
                    break

        # load human annotations
        for story_name, sub_id, _, _ in story_recall_segments:
            human_ratings_dict[story_name][sub_id] = (
                load_cyoa_recall_matrix_human_binary(
                    story_name=story_name, sub_id=sub_id
                )
            )

    elif testset.startswith("memsearch"):
        story_names_memsearch = [
            "ednora",
            "hollow",
            "i_love_death",
        ]
        story_recall_segments = list()
        for story_name in story_names_memsearch:
            story_recall_segments_single, _, _ = load_story_recall_segments(
                story_name=story_name,
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            if not story_recall_segments_single:
                raise ValueError(
                    f"No story-recall segments found for story {story_name!r}."
                )
            story_recall_segments.append((story_name, *story_recall_segments_single[0]))

        # load human annotations
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
                human_ratings_dict[story_name][sub_id] = rm_memsearch
    else:
        raise ValueError(f"Invalid testset: {testset}")

    return story_recall_segments, human_ratings_dict


def get_average_pairwise_f1(
    recall_matrices_dct: dict[str, list[np.ndarray]],
) -> float:
    f1_scores = []
    for recall_matrices in recall_matrices_dct.values():
        flat_matrices = [m.flatten() for m in recall_matrices]
        scores = []

        for m_i, m_j in combinations(flat_matrices, 2):
            score = f1_score(m_i, m_j, average="binary")
            scores.append(score)

        if scores:
            f1_scores.append(np.mean(scores))

    if not f1_scores:
        return 0.0
    return np.mean(f1_scores).item()


def get_krippendorff_alpha(recall_matrices_dct: dict[str, list[np.ndarray]]) -> float:
    """Return the agreement counting each individual recall equally.

    This measures how consistent ratings are for each individual recall.
    E.g. if a story contributes 50 recalls and another only 10, the first story
    contributes 5 times as much to the agreement as the second story.
    This is okay, because we are interested in the consistency of the rater
    at the level of individual recalls. It also probably doesn't matter to average
    across stories instead.
    """
    pooled_data = []

    for matrices in recall_matrices_dct.values():
        subject_flattened = np.array([m.flatten() for m in matrices])
        pooled_data.append(subject_flattened)
    pooled_data = np.concatenate(pooled_data, axis=1)

    alpha = krippendorff.alpha(
        reliability_data=pooled_data, level_of_measurement="nominal"
    )
    return alpha


def evaluate_repeat_reliability(
    n_repeats: int,
    rater_name: str,
    model_name: str | None,
    testset: str,
    device: str | None = None,
    seed: int = 42,
    random_mode: str | None = None,
    window_size: int = 5,
    dry_run: bool = False,
    movie_mode: bool = False,
    reranker_threshold: float | None = None,
    top_k: int = 5,
):
    """Evaluate the repeat reliability of the rater."""
    rater = initialize_rater(
        rater_name=rater_name,
        model_name=model_name,
        device=device,
        window_size=window_size,
        dry_run=dry_run,
        reranker_threshold=reranker_threshold,
        top_k=top_k,
        movie_mode=movie_mode,
    )
    if hasattr(rater, "model_name"):
        model_name = rater.model_name  # type: ignore
    else:
        model_name = None

    output_dir = (
        Path("data")
        / "eval"
        / eval_param_str(
            repeat_reliability=True,
            testset=testset,
            rater_name=rater_name,
            model_name=model_name,
            seed=seed,
            random_mode=random_mode,
        )
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    story_recall_segments, human_ratings_dict = (
        load_story_recall_segments_repeat_reliability(
            testset=testset,
        )
    )

    rng = np.random.default_rng(seed)

    precisions: dict[str, list] = defaultdict(list)
    recalls: dict[str, list] = defaultdict(list)
    pearsonrs: dict[str, list] = defaultdict(list)
    f1s: dict[str, list] = defaultdict(list)
    recall_matrices_model_dct: dict[str, list[np.ndarray]] = defaultdict(list)
    recall_matrices_comparison_dct: dict[str, np.ndarray] = dict()
    for (
        story_name,
        sub_id,
        story_segments,
        recall_segments,
    ) in tqdm(story_recall_segments, desc="(eval)"):
        recall_id = f"{story_name}_{sub_id}"

        # a) get ground truth
        rm_comparison = human_ratings_dict[story_name][sub_id]  # type: ignore
        recall_matrices_comparison_dct[recall_id] = rm_comparison

        if (rm_comparison == 0).all():
            raise ValueError(
                f"Comparison matrix is all zero for {story_name=} {sub_id=}"
                " choose different recall"
            )

        # b) get model ratings
        for _ in range(n_repeats):
            rm_model = get_model_ratings(
                random_mode=random_mode,
                rng=rng,
                rm_comparison=rm_comparison,
                rater=rater,
                story_segments=story_segments,
                recall_segments=recall_segments,
            )
            recall_matrices_model_dct[recall_id].append(rm_model)

            if (rm_model == 0).all() or dry_run:
                precision = 0
                recall = 0
                pearsonr_score = 0
            else:
                rm_model_flat = rm_model.flatten()
                rm_comparison_flat = rm_comparison.flatten()
                precision = precision_score(rm_comparison_flat, rm_model_flat)
                recall = recall_score(rm_comparison_flat, rm_model_flat)
                pearsonr_score: float = pearsonr(rm_comparison_flat, rm_model_flat)[0]  # type: ignore

            precisions[recall_id].append(precision)
            recalls[recall_id].append(recall)
            pearsonrs[recall_id].append(pearsonr_score)
            denom = precision + recall
            f1 = (2 * precision * recall) / denom if denom != 0 else 0.0
            f1s[recall_id].append(f1)

    if dry_run:
        console.print(f"[DRY RUN] Estimated Usage: {rater.get_usage()}")
        return

    # output results
    if random_mode is not None:
        rater_str = random_mode
    else:
        rater_str = rater_name
    console.print(f"Rater: {rater_str} | N recalls: {len(story_recall_segments)}")

    mean_f1s = list()
    mean_precisions = list()
    mean_recalls = list()
    mean_pearsonrs = list()
    for recall_id in f1s.keys():
        mean_f1_score = np.mean(f1s[recall_id])
        std_f1_score = np.std(f1s[recall_id])
        mean_precision = np.mean(precisions[recall_id])
        std_precision = np.std(precisions[recall_id])
        mean_recall = np.mean(recalls[recall_id])
        std_recall = np.std(recalls[recall_id])
        mean_pearsonr = np.mean(pearsonrs[recall_id])
        std_pearsonr = np.std(pearsonrs[recall_id])
        console.print(
            f"[yellow]{recall_id}[/yellow]"
            f"\n mean pearsonr={mean_pearsonr:.3f} ({std_pearsonr:.3f})"
            f"\n mean f1={mean_f1_score:.3f} ({std_f1_score:.3f})"
            f"\n mean precision={mean_precision:.3f} ({std_precision:.3f})"
            f"\n mean recall={mean_recall:.3f} ({std_recall:.3f})"
        )
        mean_f1s.append(mean_f1_score)
        mean_precisions.append(mean_precision)
        mean_recalls.append(mean_recall)
        mean_pearsonrs.append(mean_pearsonr)

    overall_mean_f1 = np.mean(mean_f1s)
    overall_mean_precision = np.mean(mean_precisions)
    overall_mean_recall = np.mean(mean_recalls)
    overall_mean_pearsonr = np.mean(mean_pearsonrs)

    console.print(f"\nRater: {rater_str} | N recalls: {len(story_recall_segments)}")
    console.print(
        f"[yellow]Overall[/yellow] (compared to human annotations)"
        f"\n mean pearsonr={overall_mean_pearsonr:.3f}"
        f"\n mean f1={overall_mean_f1:.3f}"
        f", mean precision={overall_mean_precision:.3f}"
        f", mean recall={overall_mean_recall:.3f}"
    )

    # compare repeat reliability with itself
    mean_pairwise_f1 = get_average_pairwise_f1(recall_matrices_model_dct)
    kripp_alpha = get_krippendorff_alpha(recall_matrices_model_dct)
    console.print(
        f"\n[yellow]Overall[/yellow] (compared to itself)"
        f"\n mean pairwise f1={mean_pairwise_f1:.3f}"
        f"\n krippendorff alpha={kripp_alpha:.3f}"
    )

    results_dict = {
        "testset": testset,
        "rater_name": rater_name,
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "random_mode": random_mode,
        "n_repeats": n_repeats,
        "overall_mean_f1": float(overall_mean_f1),
        "overall_mean_precision": float(overall_mean_precision),
        "overall_mean_recall": float(overall_mean_recall),
        "overall_mean_pearsonr": float(overall_mean_pearsonr),
        "mean_pairwise_f1": float(mean_pairwise_f1),
        "krippendorff_alpha": float(kripp_alpha),
        "f1s": {k: [float(x) for x in v] for k, v in f1s.items()},
        "precisions": {k: [float(x) for x in v] for k, v in precisions.items()},
        "recalls": {k: [float(x) for x in v] for k, v in recalls.items()},
        "pearsonrs": {k: [float(x) for x in v] for k, v in pearsonrs.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)  # make again, in case user deleted it
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    console.print(f"Saved results to {results_path}")

    # save recall matrices
    recall_matrices_model_path = output_dir / "recall_matrices_model_dct.pkl"
    recall_matrices_comparison_path = output_dir / f"recall_matrices_{testset}_dct.pkl"

    with open(recall_matrices_model_path, "wb") as f:
        pickle.dump(recall_matrices_model_dct, f)
    with open(recall_matrices_comparison_path, "wb") as f:
        pickle.dump(recall_matrices_comparison_dct, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-R",
        "--repeat_reliability",
        action="store_true",
        default=False,
        help=(
            "Whether to evaluate the repeat reliability."
            " If True, will evaluate 3 recalls within the testset 10 times."
        ),
    )
    args.add_argument(
        "-r",
        "--rater_name",
        choices=["reranker", "openai", "huggingface", "anthropic"],
        default="openai",
        help="Name of the rater to use. Default is 'openai'.",
    )
    args.add_argument(
        "-t",
        "--testset",
        choices=[
            "cyoa",
            "cyoa_alice10",
            "cyoa_monthiversary6",
            "memsearch",
            "memsearch10",
        ],
        default="cyoa_alice10",
        help=("Name of the testset. Default is 'cyoa_alice10'."),
    )
    args.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=None,
        help=("[reranker, openai, huggingface] Name of the model to use."),
    )
    args.add_argument(
        "--device",
        type=str,
        default=None,
        help=("[reranker, huggingface] Device to use. If None, will be autoselected."),
    )
    args.add_argument(
        "--random_mode",
        choices=["full_shuffle", "row_shuffle"],
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
    args.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Estimate cost without calling the API",
    )

    args.add_argument(
        "--movie",
        action="store_true",
        default=False,
        help="runs with movie-specific prompt",
    )
    args.add_argument(
        "--window_size", type=int, default=5, help="Size of recall context window (+/-)"
    )
    args.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default is 42.",
    )
    args.add_argument(
        "-n",
        "--n_repeats",
        type=int,
        default=10,
        help="[repeat_reliability] Number of times to run each recall. Default is 10.",
    )
    args = args.parse_args()

    if args.repeat_reliability:
        evaluate_repeat_reliability(
            n_repeats=args.n_repeats,
            rater_name=args.rater_name,
            testset=args.testset,
            model_name=args.model_name,
            device=args.device,
            seed=args.seed,
            random_mode=args.random_mode,
            window_size=args.window_size,
            dry_run=args.dry_run,
            movie_mode=args.movie,
            reranker_threshold=args.reranker_threshold,
            top_k=args.top_k,
        )
    else:
        evaluate(
            repeat_reliability=False,
            rater_name=args.rater_name,
            testset=args.testset,
            model_name=args.model_name,
            device=args.device,
            seed=args.seed,
            random_mode=args.random_mode,
            window_size=args.window_size,
            dry_run=args.dry_run,
            movie_mode=args.movie,
            reranker_threshold=args.reranker_threshold,
            top_k=args.top_k,
        )
