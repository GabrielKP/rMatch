"""
dataset_stats.py

Prints summary statistics for alice10, monthiversary6, and memsearch10:
  - number of stories
  - average story length in words
  - min and max story word counts
  - number of recalls (subjects)
  - average recall length in words (mean over all recalls in the set)
  - min and max recall word counts
"""

import numpy as np

from recall_matrix.load import (
    load_cyoa_story_recall_segments,
    load_story_recall_segments,
)

# ── dataset definitions (mirrors eval script) ─────────────────────────────────

DATASETS = {
    "alice10": {
        "type": "cyoa",
        "story_names": [
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
        ],
    },
    "monthiversary6": {
        "type": "cyoa",
        "story_names": [
            "monthiversary_3",
            "monthiversary_4",
            "monthiversary_10",
            "monthiversary_14",
            "monthiversary_19",
            "monthiversary_23",
            "monthiversary_25",
        ],
    },
    "memsearch10": {
        "type": "memsearch",
        "story_names": [
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
        ],
    },
}


def word_count(segments: list[str]) -> int:
    return sum(len(s.split()) for s in segments)


def load_segments(
    dataset_name: str, cfg: dict
) -> list[tuple[str, str, list[str], list[str]]]:
    """Returns list of (story_name, sub_id, story_segments, recall_segments)."""
    if cfg["type"] == "cyoa":
        return load_cyoa_story_recall_segments(story_names=cfg["story_names"])

    # memsearch
    rows = []
    for story_name in cfg["story_names"]:
        single, _, _ = load_story_recall_segments(
            story_name=story_name,
            story_segmentation_method="seg_c",
            recall_segmentation_method="sentences",
        )
        for sub_id, story_segs, recall_segs in single:
            rows.append((story_name, sub_id, story_segs, recall_segs))
    return rows


def compute_stats(dataset_name: str, cfg: dict) -> None:
    rows = load_segments(dataset_name, cfg)

    # story lengths: one entry per unique story (segments are the same across subjects)
    story_words: dict[str, int] = {}
    for story_name, _, story_segs, _ in rows:
        if story_name not in story_words:
            story_words[story_name] = word_count(story_segs)

    recall_word_counts = [word_count(recall_segs) for _, _, _, recall_segs in rows]

    n_stories = len(story_words)
    story_word_list = list(story_words.values())
    avg_story_len = float(np.mean(story_word_list))
    min_story_len = min(story_word_list)
    max_story_len = max(story_word_list)

    n_recalls = len(rows)
    avg_recall_len = float(np.mean(recall_word_counts))
    min_recall_len = min(recall_word_counts)
    max_recall_len = max(recall_word_counts)

    print(f"\n{'=' * 42}")
    print(f"  {dataset_name}")
    print(f"{'=' * 42}")
    print(f"  Stories              : {n_stories}")
    print(f"  Avg story length     : {avg_story_len:.0f} words")
    print(f"  Story length range   : {min_story_len}-{max_story_len} words")
    print(f"  Recalls (subjects)   : {n_recalls}")
    print(f"  Avg recall length    : {avg_recall_len:.0f} words")
    print(f"  Recall length range  : {min_recall_len}-{max_recall_len} words")


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        compute_stats(name, cfg)
    print()
