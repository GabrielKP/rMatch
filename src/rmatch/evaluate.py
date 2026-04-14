import argparse
import datetime
import json
import os
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
from tqdm import tqdm

from rmatch import console, matchlist_type
from rmatch.matchers.matcher import Matcher
from rmatch.utils import (
    atomic_write_json,
    binary_f1,
    binary_precision,
    binary_recall,
    match_list_to_matrix,
    pearsonr,
)

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
    env_path = os.environ.get("BENCHMARK_ROOT")
    if env_path is not None:
        return Path(env_path)
    # you can always try it...
    maybe_path = Path("../rBench")
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
    dict[str, tuple[np.ndarray, matchlist_type]],
]:
    """Return story-recall segments and human matches of one story.

    Returns:
    - story-recall segments: list[tuple[str, str, list[str], list[str]]]
    - human matches: dict[str, tuple[np.ndarray, list[tuple[int, list[int]]]]]
        - sub_id -> (human_matches_matrix, match_list)
            - human_matches_matrix: np.ndarray
            - match_list: list[tuple[int, list[int]]]
                - story_segment_idx -> [matched_segment_idx, ...]
    """
    t_method, r_method, matches_name = _SEGMENTS[testset]
    story_dir = benchmark_root / "data" / testset / story_name

    with open(story_dir / "transcripts" / f"{t_method}.json") as f:
        tj = json.load(f)
    story_segments = tj["segments"]

    with open(story_dir / "recalls" / f"{r_method}.json") as f:
        rj = json.load(f)
    recalls_map: dict[str, list[str]] = rj["recalls"]

    with open(story_dir / "matches" / matches_name) as f:
        match_json = json.load(f)
    n_story = match_json["n_story_segments"]
    # sub_id -> [story_segment_idx, [matched_segment_idx, ...]]
    ratings: dict[str, matchlist_type] = match_json["ratings"]

    human_matches_dct: dict[str, tuple[np.ndarray, matchlist_type]] = {}
    for sub_id, match_list in ratings.items():
        human_matches_dct[sub_id] = (
            match_list_to_matrix(match_list, n_story),
            match_list,
        )

    rows: list[tuple[str, str, list[str], list[str]]] = []
    for sub_id in sorted(recalls_map.keys()):
        if sub_id not in ratings:
            continue
        rows.append(
            (story_name, sub_id, story_segments, recalls_map[sub_id]),
        )

    return rows, human_matches_dct


def load_benchmark_full_eval(
    benchmark_root: Path,
    testset: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, dict[str, tuple[np.ndarray, matchlist_type]]],
]:
    if testset not in _SEGMENTS:
        raise ValueError(f"Invalid testset: {testset}")

    dataset_dir = benchmark_root / "data" / testset
    stories = load_dataset_stories(dataset_dir)

    human_ratings_dict: dict[str, dict[str, tuple[np.ndarray, matchlist_type]]] = (
        defaultdict(dict)
    )
    story_recall_segments: list[tuple[str, str, list[str], list[str]]] = []

    for story_name in stories:
        rows, human_matches_dct = _load_one_story(benchmark_root, testset, story_name)
        for sub_id, (rm, match_list) in human_matches_dct.items():
            human_ratings_dict[story_name][sub_id] = (rm, match_list)
        story_recall_segments.extend(rows)

    return story_recall_segments, human_ratings_dict


def load_benchmark_repeat_reliability(
    benchmark_root: Path,
    testset: str,
) -> tuple[
    list[tuple[str, str, list[str], list[str]]],
    dict[str, dict[str, tuple[np.ndarray, matchlist_type]]],
]:
    if testset not in _REPEAT_RELIABILITY_STORIES:
        raise ValueError(
            f"Invalid testset for repeat reliability: {testset}. "
            f"Expected one of {list(_REPEAT_RELIABILITY_STORIES)}."
        )

    story_names = _REPEAT_RELIABILITY_STORIES[testset]
    human_ratings_dict: dict[str, dict[str, tuple[np.ndarray, matchlist_type]]] = (
        defaultdict(dict)
    )
    story_recall_segments: list[tuple[str, str, list[str], list[str]]] = []

    for story_name in story_names:
        rows, human_matches_dct = _load_one_story(benchmark_root, testset, story_name)
        for sub_id, (rm, match_list) in human_matches_dct.items():
            human_ratings_dict[story_name][sub_id] = (rm, match_list)
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


def save_raw_response(
    raw_response_dir: Path,
    raw_response_incorrect_dir: Path,
    story_name: str,
    sub_id: str,
    recall_segment_idx: int,
    repeat_idx: int,
    attempt: int,
    human_response: list[int],
    parsed_response_set: set[int] | None,
    prompt: str,
    response: str,
    pearsonr: float,
    f1: float,
    precision: float,
    recall: float,
):
    if parsed_response_set is None:
        is_correct = False
        parsed_response_str = "PARSED RESPONSE: < parsing failed >"
    else:
        model_response = list(parsed_response_set)
        is_correct = (len(human_response) == len(model_response)) and (
            all(
                int(x) == int(y)
                for x, y in zip(
                    sorted(model_response),
                    sorted(human_response),
                )
            )
        )
        parsed_response_str = (
            f"PARSED RESPONSE: {','.join(str(x) for x in model_response)}"
        )

    desc = (
        f"{story_name} - {sub_id} - repeat: {repeat_idx} - recall idx:"
        f" {recall_segment_idx:03d} - attempt {attempt}"
    )
    separator1 = "-" * len(desc)
    correct = "-- CORRECT --\n" if is_correct else "-- INCORRECT --"
    correct_response_str = f"HUMAN RESPONSE: {','.join(str(x) for x in human_response)}"
    story_metrics_str = "STORY METRICS (entire recall):\n"
    pearsonr_str = f"PEARSONR: {pearsonr:.3f}"
    f1_str = f"F1: {f1:.3f}"
    precision_str = f"PRECISION: {precision:.3f}"
    recall_str = f"RECALL: {recall:.3f}"
    separator2 = "\n------------------------------------------------\n"
    prompt_and_reponse = f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}"

    output_text = "\n".join(
        [
            desc,
            separator1,
            correct,
            correct_response_str,
            parsed_response_str,
            story_metrics_str,
            pearsonr_str,
            f1_str,
            precision_str,
            recall_str,
            separator2,
            prompt_and_reponse,
        ]
    )

    file_sub_path = (
        Path(story_name)
        / sub_id
        / (
            f"recall_repeat{repeat_idx:03d}_rec-seg-{recall_segment_idx:03d}_a{attempt}.txt"
        )
    )
    output_path = raw_response_dir / file_sub_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text + "\n", encoding="utf-8")

    if not is_correct:
        output_incorrect_path = raw_response_incorrect_dir / file_sub_path
        output_incorrect_path.parent.mkdir(parents=True, exist_ok=True)
        output_incorrect_path.write_text(output_text + "\n", encoding="utf-8")


def _compute_pair_metrics(
    rm_model: np.ndarray,
    rm_comparison: np.ndarray,
    zero_out: bool = False,
) -> dict[str, float]:
    """Compute precision, recall, pearsonr, accuracy, f1 for a pair."""
    if (rm_model == 0).all() or zero_out:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "pearsonr": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
        }
    flat_model = rm_model.flatten()
    flat_comp = rm_comparison.flatten()
    p = binary_precision(flat_comp, flat_model)
    r = binary_recall(flat_comp, flat_model)
    pr = float(pearsonr(flat_comp, flat_model))
    acc = float(accuracy(flat_comp, flat_model))
    denom = p + r
    f1 = (2 * p * r) / denom if denom != 0 else 0.0
    return {"precision": p, "recall": r, "pearsonr": pr, "accuracy": acc, "f1": f1}


def get_average_pairwise_f1(
    recall_matrices_dct: dict[str, list[np.ndarray]],
) -> float:
    f1_scores = []
    for recall_matrices in recall_matrices_dct.values():
        flat_matrices = [m.flatten() for m in recall_matrices]
        scores = []

        for m_i, m_j in combinations(flat_matrices, 2):
            score = binary_f1(m_i, m_j)
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


def evaluate(
    testset: str,
    benchmark_root: Path,
    matcher_name: str,
    repeat_reliability: bool = False,
    n_repeats: int = 5,
    dry_run: bool = False,
    **kwargs,
):
    """Evaluate matcher on testset.

    When repeat_reliability=True, each recall is matched n_repeats times
    to measure self-consistency (Krippendorff alpha, pairwise F1).
    """
    if not repeat_reliability:
        n_repeats = 1

    matcher_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    matcher = Matcher(matcher_name=matcher_name, **matcher_kwargs)  # type: ignore[call-arg]
    model_name = getattr(matcher, "model_name", None)

    output_dir = (
        Path("data")
        / "eval"
        / eval_param_str(
            repeat_reliability=repeat_reliability,
            testset=testset,
            matcher_name=matcher_name,
            model_name=model_name,
        )
    )
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if repeat_reliability:
        story_recall_segments, human_ratings_dict = load_benchmark_repeat_reliability(
            benchmark_root, testset
        )
    else:
        story_recall_segments, human_ratings_dict = load_benchmark_full_eval(
            benchmark_root, testset
        )

    # Metrics keyed by recall_id; list length is 1 (full eval) or n_repeats (RR).
    precisions: dict[str, list[float]] = defaultdict(list)
    recalls_metric: dict[str, list[float]] = defaultdict(list)
    pearsonrs: dict[str, list[float]] = defaultdict(list)
    accuracies: dict[str, list[float]] = defaultdict(list)
    f1s: dict[str, list[float]] = defaultdict(list)
    recall_matrices_model_dct: dict[str, list[np.ndarray]] = defaultdict(list)
    recall_matrices_human_dct: dict[str, np.ndarray] = {}
    n_skipped = 0
    last_story_name: str | None = None
    last_sub_id: str | None = None
    last_repeat_index: int | None = None

    def _pearson_json(x: float) -> float | None:
        xf = float(x)
        return None if np.isnan(xf) else xf

    def _save_checkpoint(*, reason: str | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        body: dict = {
            "checkpoint": True,
            "testset": testset,
            "matcher_name": matcher_name,
            "model_name": model_name,
            **{k: v for k, v in kwargs.items()},
        }
        if repeat_reliability:
            body["n_repeats"] = n_repeats
            body["progress"] = {
                "recall_ids_total": len(story_recall_segments),
                "last_recall_id": (
                    f"{last_story_name}_{last_sub_id}"
                    if last_story_name is not None
                    else None
                ),
                "last_completed_repeat_index": last_repeat_index,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **({"reason": reason} if reason is not None else {}),
            }
            body["f1s"] = {k: [float(x) for x in v] for k, v in f1s.items()}
            body["precisions"] = {
                k: [float(x) for x in v] for k, v in precisions.items()
            }
            body["recalls"] = {
                k: [float(x) for x in v] for k, v in recalls_metric.items()
            }
            body["pearsonrs"] = {
                k: [_pearson_json(x) for x in v] for k, v in pearsonrs.items()
            }
        else:
            all_p = [v for vals in precisions.values() for v in vals]
            all_r = [v for vals in recalls_metric.values() for v in vals]
            all_pr = [v for vals in pearsonrs.values() for v in vals]
            all_acc = [v for vals in accuracies.values() for v in vals]
            body["progress"] = {
                "total_items": len(story_recall_segments),
                "evaluated_count": len(all_p),
                "skipped_all_zero_count": n_skipped,
                "last_story_name": last_story_name,
                "last_sub_id": last_sub_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **({"reason": reason} if reason is not None else {}),
            }
            body["precisions"] = [float(x) for x in all_p]
            body["recalls"] = [float(x) for x in all_r]
            body["pearsonrs"] = [_pearson_json(x) for x in all_pr]
            body["accuracies"] = [float(x) for x in all_acc]

        if matcher.get_usage() is not None:
            body["usage"] = matcher.get_usage()
        atomic_write_json(output_dir / "checkpoint.json", body, indent=2, default=str)

        if repeat_reliability:
            with open(output_dir / "recall_matrices_model_dct.partial.pkl", "wb") as f:
                pickle.dump(dict(recall_matrices_model_dct), f)
            with open(output_dir / "recall_matrices_human_dct.partial.pkl", "wb") as f:
                pickle.dump(dict(recall_matrices_human_dct), f)
        else:
            model_list = [
                m for mats in recall_matrices_model_dct.values() for m in mats
            ]
            comp_list = list(recall_matrices_human_dct.values())
            with open(output_dir / "recall_matrices_model.partial.pkl", "wb") as f:
                pickle.dump(model_list, f)
            with open(output_dir / "recall_matrices_human.partial.pkl", "wb") as f:
                pickle.dump(comp_list, f)

    # ---- eval loop ----
    # match_tracker is a list of tuples:
    # - story_name
    # - sub_id
    # - repeat_idx
    # - is_correct
    # - parsed_response
    # - correct_response
    # - story: (pearsonr, f1, precision, recall)
    raw_response_dir = output_dir / "raw"
    raw_response_incorrect_dir = output_dir / "raw_incorrect"
    try:
        for (
            story_name,
            sub_id,
            story_segments,
            recall_segments,
        ) in tqdm(story_recall_segments, desc="(eval)"):
            recall_id = f"{story_name}_{sub_id}"
            rm_human, match_list_human = human_ratings_dict[story_name][sub_id]  # type: ignore
            recall_matrices_human_dct[recall_id] = rm_human

            if (rm_human == 0).all():
                if repeat_reliability:
                    raise ValueError(
                        f"Comparison matrix is all zero for {story_name=}"
                        f" {sub_id=} choose different recall"
                    )
                console.print(
                    f"Skipping {story_name=} {sub_id=} : comparison matrix is all zero"
                )
                n_skipped += 1
                continue

            for repeat_i in range(n_repeats):
                match_key = f"{story_name}_{sub_id}_{repeat_i}"
                match_list_model = matcher.match(
                    story_segments=story_segments,
                    recall_segments=recall_segments,
                    match_key=match_key,
                )

                rm_model = match_list_to_matrix(
                    match_list_model,
                    len(story_segments),
                )
                recall_matrices_model_dct[recall_id].append(rm_model)

                metrics = _compute_pair_metrics(
                    rm_model,
                    rm_human,
                    zero_out=(not repeat_reliability and dry_run),
                )
                precisions[recall_id].append(metrics["precision"])
                recalls_metric[recall_id].append(metrics["recall"])
                pearsonrs[recall_id].append(metrics["pearsonr"])
                accuracies[recall_id].append(metrics["accuracy"])
                f1s[recall_id].append(metrics["f1"])

                # save raw responses
                prompt_responses = matcher.prompt_response_log[match_key]
                for recall_segment_idx in range(len(recall_segments)):
                    human_response = match_list_human[recall_segment_idx][1]
                    for attempt, (prompt, response, parsed_response_set) in enumerate(
                        prompt_responses[recall_segment_idx]
                    ):
                        save_raw_response(
                            raw_response_dir,
                            raw_response_incorrect_dir,
                            story_name=story_name,
                            sub_id=sub_id,
                            recall_segment_idx=recall_segment_idx,
                            repeat_idx=repeat_i,
                            attempt=attempt,
                            human_response=human_response,
                            parsed_response_set=parsed_response_set,
                            prompt=prompt,
                            response=response,
                            pearsonr=metrics["pearsonr"],
                            f1=metrics["f1"],
                            precision=metrics["precision"],
                            recall=metrics["recall"],
                        )

                last_story_name = story_name
                last_sub_id = sub_id
                last_repeat_index = repeat_i
                _save_checkpoint()
    except KeyboardInterrupt:
        _save_checkpoint(reason="KeyboardInterrupt")
        console.print(
            f"[yellow]Interrupted; checkpoint written under[/yellow] {output_dir}"
        )
        raise

    if dry_run:
        console.print(f"[DRY RUN] Estimated Usage: {matcher.get_usage()}")
        return

    # ---- results ----
    output_dir.mkdir(parents=True, exist_ok=True)

    if repeat_reliability:
        console.print(
            f"Matcher: {matcher_name} | N recalls: {len(story_recall_segments)}"
        )

        mean_f1s = []
        mean_precisions = []
        mean_recalls = []
        mean_pearsonrs = []
        for recall_id in f1s:
            mean_f1 = np.mean(f1s[recall_id])
            std_f1 = np.std(f1s[recall_id])
            mean_precision = np.mean(precisions[recall_id])
            std_precision = np.std(precisions[recall_id])
            mean_recall = np.mean(recalls_metric[recall_id])
            std_recall = np.std(recalls_metric[recall_id])
            mean_pearsonr = np.mean(pearsonrs[recall_id])
            std_pearsonr = np.std(pearsonrs[recall_id])
            console.print(
                f"[yellow]{recall_id}[/yellow]"
                f"\n mean pearsonr={mean_pearsonr:.3f} ({std_pearsonr:.3f})"
                f"\n mean f1={mean_f1:.3f} ({std_f1:.3f})"
                f"\n mean precision={mean_precision:.3f} ({std_precision:.3f})"
                f"\n mean recall={mean_recall:.3f} ({std_recall:.3f})"
            )
            mean_f1s.append(mean_f1)
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

        mean_pairwise_f1 = get_average_pairwise_f1(recall_matrices_model_dct)
        kripp_alpha = get_krippendorff_alpha(recall_matrices_model_dct)
        console.print(
            f"\n[yellow]Overall[/yellow] (compared to itself)"
            f"\n mean pairwise f1={mean_pairwise_f1:.3f}"
            f"\n krippendorff alpha={kripp_alpha:.3f}"
        )

        results_dict: dict = {
            "testset": testset,
            "matcher_name": matcher_name,
            "model_name": model_name,
            **kwargs,
            "n_repeats": n_repeats,
            "overall_mean_f1": float(overall_mean_f1),
            "overall_mean_precision": float(overall_mean_precision),
            "overall_mean_recall": float(overall_mean_recall),
            "overall_mean_pearsonr": float(overall_mean_pearsonr),
            "mean_pairwise_f1": float(mean_pairwise_f1),
            "krippendorff_alpha": float(kripp_alpha),
            "f1s": {k: [float(x) for x in v] for k, v in f1s.items()},
            "precisions": {k: [float(x) for x in v] for k, v in precisions.items()},
            "recalls": {k: [float(x) for x in v] for k, v in recalls_metric.items()},
            "pearsonrs": {k: [float(x) for x in v] for k, v in pearsonrs.items()},
        }

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        console.print(f"Saved results to {results_path}")

        with open(output_dir / "recall_matrices_model_dct.pkl", "wb") as f:
            pickle.dump(dict(recall_matrices_model_dct), f)
        with open(output_dir / f"recall_matrices_{testset}_dct.pkl", "wb") as f:
            pickle.dump(dict(recall_matrices_human_dct), f)

    else:
        all_p = [v for vals in precisions.values() for v in vals]
        all_r = [v for vals in recalls_metric.values() for v in vals]
        all_pr = [v for vals in pearsonrs.values() for v in vals]
        all_acc = [v for vals in accuracies.values() for v in vals]

        console.print(
            f"Matcher: {matcher_name} | N recalls: {len(all_p)}"
            f" (evaluated) / {len(story_recall_segments)} (total)"
        )

        if not all_p:
            raise ValueError(
                "No recalls were evaluated (all comparison matrices were "
                "all-zero or empty)."
            )

        precision_macro = np.mean(all_p)
        recall_macro = np.mean(all_r)
        denom = precision_macro + recall_macro
        f1_macro = (2 * precision_macro * recall_macro) / denom if denom != 0 else 0.0
        console.print(
            f"Macro: F1={f1_macro:.3f},"
            f" Precision={precision_macro:.3f},"
            f" Recall={recall_macro:.3f}"
        )

        accuracy_macro = np.mean(all_acc)
        console.print(f"Accuracy: {accuracy_macro:.3f}")

        pearsonr_macro = np.mean(all_pr)
        console.print(f"Pearsonr: {pearsonr_macro:.3f}")

        results_dict = {
            "testset": testset,
            "matcher_name": matcher_name,
            **kwargs,
            "f1_macro": float(f1_macro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "accuracy_macro": float(accuracy_macro),
            "pearsonr_macro": float(pearsonr_macro),
        }

        if matcher.get_usage() is not None:
            console.print(f"Total API usage: {matcher.get_usage()}")
            results_dict["usage"] = matcher.get_usage()

        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        console.print(f"Saved results to {results_path}")

        model_list = [m for mats in recall_matrices_model_dct.values() for m in mats]
        comp_list = list(recall_matrices_human_dct.values())
        with open(output_dir / "recall_matrices_model.pkl", "wb") as f:
            pickle.dump(model_list, f)
        with open(output_dir / "recall_matrices_human.pkl", "wb") as f:
            pickle.dump(comp_list, f)


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
        help="[openai, huggingface] Name of the model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="[huggingface] Device to use. If omitted, autoselected.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Size of recall context window (+/-).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="[anthropic, openai] Estimate cost without calling the API.",
    )
    parser.add_argument(
        "-n",
        "--n-repeats",
        dest="n_repeats",
        type=int,
        default=None,
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
        default=None,
        help="[huggingface] Batch size.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="[huggingface] max_new_tokens for the matcher.",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        default=None,
        help="[huggingface] Print verbose errors.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        default=None,
        help="[huggingface] Disable flash-attn for the model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=[
            "primary",
            "primary_no_story",
            "primary_no_cot",
            "primary_no_story_no_cot",
            "secondary",
        ],
        default=None,
        help="[anthropic, openai, huggingface] Prompt type. Default is 'primary'.",
    )
    args = parser.parse_args()
    benchmark_root = args.benchmark_root or default_benchmark_root()

    evaluate(
        matcher_name=args.matcher,
        testset=args.testset,
        benchmark_root=benchmark_root,
        repeat_reliability=args.repeat_reliability,
        n_repeats=args.n_repeats or 5,
        model_name=args.model_name,
        device=args.device,
        window_size=args.window_size,
        dry_run=args.dry_run,
        verbose_errors=args.verbose_errors,
        quantization=args.quantization,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        no_flash_attn=args.no_flash_attn,
    )
