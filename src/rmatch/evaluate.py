import argparse
import datetime
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

import krippendorff
import numpy as np
from codecarbon import EmissionsTracker
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from rmatch import ENV, console
from rmatch.matchers import initialize_matcher
from rmatch.matchers.matcher import Matcher
from rmatch.utils import ratings_single_sub_to_matrix

# transcript stem, recall JSON stem (without .json), matches filename
_SEGMENTS: dict[str, tuple[str, str, str]] = {
    "alice": ("scenes", "sub_sentences", "human-scenes-sub_sentences.json"),
    "monthiversary": ("scenes", "sub_sentences", "human-scenes-sub_sentences.json"),
    "memsearch": ("scenes", "sentences", "human-scenes-sentences.json"),
}

_REPEAT_RELIABILITY_STORIES: dict[str, list[str]] = {
    "alice": ["alice_3", "alice_6", "alice_12"],
    "monthiversary": [
        "monthiversary_3",
        "monthiversary_4",
        "monthiversary_25",
    ],
    "memsearch": ["ednora", "hollow", "i_love_death"],
}


def default_benchmark_root() -> Path:
    env_path = ENV.get("BENCHMARK_ROOT")
    if env_path is not None:
        return Path(env_path)
    # you can always try it...
    maybe_path = Path("../benchmark")
    if maybe_path.exists():
        return maybe_path
    else:
        raise FileNotFoundError(
            f"BENCHMARK_ROOT not found at {maybe_path}."
            ' Set it in .env as BENCHMARK_ROOT="...".'
        )


def load_dataset_stories(dataset_dir: Path) -> list[str]:
    with open(dataset_dir / "dataset.json") as f:
        meta = json.load(f)
    return list(meta["stories"])


def _load_one_story(
    benchmark_root: Path,
    testset: str,
    story_name: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, np.ndarray],
]:
    t_method, r_method, matches_name = _SEGMENTS[testset]
    story_dir = benchmark_root / "data" / testset / story_name

    with open(story_dir / "transcripts" / f"{t_method}.json") as f:
        tj = json.load(f)
    story_segments = tj["segments"]

    with open(story_dir / "recalls" / f"{r_method}.json") as f:
        rj = json.load(f)
    recalls_map: dict[str, list[str]] = rj["recalls"]

    with open(story_dir / "matches" / matches_name) as f:
        mj = json.load(f)
    n_story = mj["n_story_segments"]
    ratings: dict[str, list] = mj["ratings"]

    human: dict[str, np.ndarray] = {}
    for sub_id, single in ratings.items():
        human[sub_id] = ratings_single_sub_to_matrix(single, n_story)

    rows: list[tuple[str, str, list[str], list[str]]] = []
    for sub_id in sorted(recalls_map.keys()):
        if sub_id not in ratings:
            continue
        rows.append(
            (story_name, sub_id, story_segments, recalls_map[sub_id]),
        )

    return rows, human


def load_benchmark_full_eval(
    benchmark_root: Path,
    testset: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, dict[str, np.ndarray]],
]:
    if testset not in _SEGMENTS:
        raise ValueError(f"Invalid testset: {testset}")

    dataset_dir = benchmark_root / "data" / testset
    stories = load_dataset_stories(dataset_dir)

    human_ratings_dict: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    story_recall_segments: list[tuple[str, str, list[str], list[str]]] = []

    for story_name in stories:
        rows, human = _load_one_story(benchmark_root, testset, story_name)
        for sub_id, rm in human.items():
            human_ratings_dict[story_name][sub_id] = rm
        story_recall_segments.extend(rows)

    return story_recall_segments, human_ratings_dict


def load_benchmark_repeat_reliability(
    benchmark_root: Path,
    testset: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, dict[str, np.ndarray]],
]:
    if testset not in _REPEAT_RELIABILITY_STORIES:
        raise ValueError(
            f"Invalid testset for repeat reliability: {testset}. "
            f"Expected one of {list(_REPEAT_RELIABILITY_STORIES)}."
        )

    story_names = _REPEAT_RELIABILITY_STORIES[testset]
    human_ratings_dict: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    story_recall_segments: list[tuple[str, str, list[str], list[str]]] = []

    for story_name in story_names:
        rows, human = _load_one_story(benchmark_root, testset, story_name)
        for sub_id, rm in human.items():
            human_ratings_dict[story_name][sub_id] = rm
        if not rows:
            raise ValueError(
                f"No story-recall segments with human ratings for story {story_name!r}."
            )
        # One recall per story: first subject in sorted order (stable).
        story_recall_segments.append(rows[0])

    return story_recall_segments, human_ratings_dict


def eval_param_str(
    repeat_reliability: bool,
    testset: str,
    matcher_name: str,
    model_name: str | None,
) -> str:
    """Get the param string for the evaluation."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rr_str = "-rr" if repeat_reliability else ""

    model_name_str = ""
    if model_name is not None:
        model_name_str = f"-m_{model_name.replace('/', '_')}"
    param_str = f"{timestamp}{rr_str}-{testset}-{matcher_name}{model_name_str}"
    return param_str


def accuracy(array_1: np.ndarray, array_2: np.ndarray) -> float:
    if len(array_1) == 0:
        return 0.0
    return np.sum(array_1 == array_2) / len(array_1)


def evaluate(
    matcher_name: str,
    model_name: str | None,
    testset: str,
    benchmark_root: Path,
    device: str | None = None,
    window_size: int = 5,
    dry_run: bool = False,
    verbose_errors: bool = False,
    quantization: Literal["4bit", "8bit"] | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    track_emissions: bool = False,
):
    """Evaluate the matcher."""
    matcher = initialize_matcher(
        matcher_name=matcher_name,
        model_name=model_name,
        device=device,
        window_size=window_size,
        dry_run=dry_run,
        verbose_errors=verbose_errors,
        quantization=quantization,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if hasattr(matcher, "model_name"):
        model_name = matcher.model_name  # type: ignore
    else:
        model_name = None

    output_dir = (
        Path("data")
        / "eval"
        / eval_param_str(
            repeat_reliability=False,
            testset=testset,
            matcher_name=matcher_name,
            model_name=model_name,
        )
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    story_recall_segments, human_ratings_dict = load_benchmark_full_eval(
        benchmark_root, testset
    )

    tracker = None
    if track_emissions:
        tracker = EmissionsTracker(
            project_name=f"rmatch-eval-{matcher_name}",
            output_dir=str(output_dir),
        )
        tracker.start()

    precisions = list()
    recalls = list()
    pearsonrs = list()
    accuracies = list()
    recall_matrices_model: list[np.ndarray] = list()
    recall_matrices_comparison: list[np.ndarray] = list()
    try:
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
            single_sub_ratings = matcher.compute_ratings_single_sub(
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
    finally:
        if tracker is not None:
            emissions_kg = tracker.stop()
            console.print(
                f"[green]Carbon emissions:[/green] {emissions_kg:.6f} kg CO2eq"
            )

    if dry_run:
        console.print(f"[DRY RUN] Estimated Usage: {matcher.get_usage()}")
        return

    # output results
    console.print(
        f"Matcher: {matcher_name} | N recalls: {len(precisions)}"
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
        "matcher_name": matcher_name,
        "model_name": model_name,
        "device": device,
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "accuracy_macro": float(accuracy_macro),
        "pearsonr_macro": float(pearsonr_macro),
    }

    if matcher.get_usage() is not None:
        console.print(f"Total API usage: {matcher.get_usage()}")
        results_dict["usage"] = matcher.get_usage()

    if track_emissions:
        results_dict["emissions_kg_co2eq"] = float(emissions_kg)  # type: ignore

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
    This is okay, because we are interested in the consistency of the matcher
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
    matcher_name: str,
    model_name: str | None,
    testset: str,
    benchmark_root: Path,
    device: str | None = None,
    window_size: int = 5,
    dry_run: bool = False,
    verbose_errors: bool = False,
    quantization: Literal["4bit", "8bit"] | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    track_emissions: bool = False,
):
    """Evaluate the repeat reliability of the matcher."""
    matcher = initialize_matcher(
        matcher_name=matcher_name,
        model_name=model_name,
        device=device,
        window_size=window_size,
        dry_run=dry_run,
        verbose_errors=verbose_errors,
        quantization=quantization,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if hasattr(matcher, "model_name"):
        model_name = matcher.model_name  # type: ignore
    else:
        model_name = None

    output_dir = (
        Path("data")
        / "eval"
        / eval_param_str(
            repeat_reliability=True,
            testset=testset,
            matcher_name=matcher_name,
            model_name=model_name,
        )
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    story_recall_segments, human_ratings_dict = load_benchmark_repeat_reliability(
        benchmark_root, testset
    )

    tracker = None
    if track_emissions:
        tracker = EmissionsTracker(
            project_name=f"rmatch-eval-rr-{matcher_name}",
            output_dir=str(output_dir),
        )
        tracker.start()

    precisions: dict[str, list] = defaultdict(list)
    recalls: dict[str, list] = defaultdict(list)
    pearsonrs: dict[str, list] = defaultdict(list)
    f1s: dict[str, list] = defaultdict(list)
    recall_matrices_model_dct: dict[str, list[np.ndarray]] = defaultdict(list)
    recall_matrices_comparison_dct: dict[str, np.ndarray] = dict()
    try:
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
                single_sub_ratings = matcher.compute_ratings_single_sub(
                    story_segments=story_segments,
                    recall_segments=recall_segments,
                    output_scores=False,
                )
                rm_model = ratings_single_sub_to_matrix(
                    single_sub_ratings,  # type: ignore
                    len(story_segments),
                )
                recall_matrices_model_dct[recall_id].append(rm_model)

                if (rm_model == 0).all():
                    precision = 0
                    recall = 0
                    pearsonr_score = 0
                else:
                    rm_model_flat = rm_model.flatten()
                    rm_comparison_flat = rm_comparison.flatten()
                    precision = precision_score(rm_comparison_flat, rm_model_flat)
                    recall = recall_score(rm_comparison_flat, rm_model_flat)
                    pearsonr_score: float = pearsonr(rm_comparison_flat, rm_model_flat)[
                        0
                    ]  # type: ignore

                precisions[recall_id].append(precision)
                recalls[recall_id].append(recall)
                pearsonrs[recall_id].append(pearsonr_score)
                denom = precision + recall
                f1 = (2 * precision * recall) / denom if denom != 0 else 0.0
                f1s[recall_id].append(f1)
    finally:
        if tracker is not None:
            emissions_kg = tracker.stop()
            console.print(
                f"[green]Carbon emissions:[/green] {emissions_kg:.6f} kg CO2eq"
            )

    if dry_run:
        console.print(f"[DRY RUN] Estimated Usage: {matcher.get_usage()}")
        return

    # output results
    console.print(f"Matcher: {matcher_name} | N recalls: {len(story_recall_segments)}")

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

    console.print(
        f"\nMatcher: {matcher_name} | N recalls: {len(story_recall_segments)}"
    )
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
        "matcher_name": matcher_name,
        "model_name": model_name,
        "device": device,
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
    if track_emissions:
        results_dict["emissions_kg_co2eq"] = float(emissions_kg)  # type: ignore
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "testset",
        type=str,
        choices=[
            "alice",
            "monthiversary",
            "memsearch",
        ],
        help="Name of the testset.",
    )

    parser.add_argument(
        "-rr",
        "--repeat-reliability",
        action="store_true",
        default=False,
        help=(
            "Evaluate repeat reliability: three fixed recalls per testset, "
            "each repeated --n-repeats times."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help=(
            "Root directory of benchmark (contains data/). "
            "Default: benchmark_ROOT environment variable, else "
            "/Users/gkressi1/opt/benchmark."
        ),
    )
    parser.add_argument(
        "-M",
        "--matcher",
        choices=[
            "anthropic",
            "reranker",
            "openai",
            "huggingface",
        ],
        default="anthropic",
        help="Name of the matcher to use. Default is 'anthropic'.",
    )

    parser.add_argument(
        "-m",
        "--model-name",
        dest="model_name",
        type=str,
        default=None,
        help="[reranker, openai, huggingface] Name of the model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="[reranker, huggingface] Device to use. If omitted, autoselected.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Estimate cost without calling the API.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Size of recall context window (+/-).",
    )
    parser.add_argument(
        "-n",
        "--n-repeats",
        dest="n_repeats",
        type=int,
        default=5,
        help="[repeat_reliability] Runs per recall. Default is 5.",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help=("[huggingface] Quantization: '4bit' or '8bit'. Default is None (bf16)."),
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=4,
        help="[huggingface] Batch size.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="[huggingface] max_new_tokens for the matcher.",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        default=False,
        help="[huggingface] Print verbose errors.",
    )
    parser.add_argument(
        "--track-emissions",
        action="store_true",
        default=False,
        help="Track carbon emissions with CodeCarbon during evaluation.",
    )
    args = parser.parse_args()
    benchmark_root = args.benchmark_root or default_benchmark_root()

    if args.repeat_reliability:
        evaluate_repeat_reliability(
            n_repeats=args.n_repeats,
            matcher_name=args.matcher,
            testset=args.testset,
            benchmark_root=benchmark_root,
            model_name=args.model_name,
            device=args.device,
            window_size=args.window_size,
            dry_run=args.dry_run,
            verbose_errors=args.verbose_errors,
            quantization=args.quantization,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            track_emissions=args.track_emissions,
        )
    else:
        evaluate(
            matcher_name=args.matcher,
            testset=args.testset,
            benchmark_root=benchmark_root,
            model_name=args.model_name,
            device=args.device,
            window_size=args.window_size,
            dry_run=args.dry_run,
            verbose_errors=args.verbose_errors,
            quantization=args.quantization,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            track_emissions=args.track_emissions,
        )
