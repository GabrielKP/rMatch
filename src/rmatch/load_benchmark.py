"""Load recall benchmark data from the benchmark JSON layout (see benchmark README)."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rmatch import ENV
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
