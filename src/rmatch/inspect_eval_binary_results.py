import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from rmatch.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
    load_ratings_dict,
    load_story_recall_segments,
)
from rmatch.utils import ratings_single_sub_to_matrix

# ── config ──────────────────────────────────────────────────────────────────
RUNS = [
    {
        "run_dir": Path(
            "data/eval/20260310_010740-memsearch10-anthropic-m_claude-haiku-4-5-seed_42"
        ),
        "testset": "memsearch10",
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
]


# ── helpers ──────────────────────────────────────────────────────────────────
def load_story_recall_segments_for_testset(testset, story_names):
    segments = []
    if testset.startswith("cyoa"):
        raw = load_cyoa_story_recall_segments(story_names=story_names)
        for story_name, sub_id, story_segs, recall_segs in raw:
            segments.append((story_name, sub_id, story_segs, recall_segs))
    elif testset.startswith("memsearch"):
        for story_name in story_names:
            single, _, _ = load_story_recall_segments(
                story_name=story_name,
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            for sub_id, story_segs, recall_segs in single:
                segments.append((story_name, sub_id, story_segs, recall_segs))
    return segments


def load_human_matrices_for_testset(
    testset, story_names, story_recall_segments
) -> dict:
    if testset.startswith("cyoa"):
        matrices = {}
        for story_name, sub_id, _, _ in story_recall_segments:
            rm = load_cyoa_recall_matrix_human_binary(
                story_name=story_name, sub_id=sub_id
            )
            matrices[(story_name, sub_id)] = rm
    elif testset.startswith("memsearch"):
        ratings_dicts = defaultdict(dict)
        for story_name in story_names:
            ratings_dict = load_ratings_dict(
                story_name=story_name,
                matcher_name="human",
                story_segmentation_method="seg_c",
                recall_segmentation_method="sentences",
            )
            n = ratings_dict["n_story_segments"]
            for sub_id, single_sub_ratings in ratings_dict["ratings"].items():
                ratings_dicts[story_name][sub_id] = ratings_single_sub_to_matrix(
                    single_sub_ratings, n
                )
        matrices = {
            (story_name, sub_id): ratings_dicts[story_name][sub_id]
            for story_name, sub_id, _, _ in story_recall_segments
            if sub_id in ratings_dicts[story_name]
        }
    else:
        raise ValueError(f"Invalid testset: {testset}")
    return matrices


def write_inspection(run_dir, testset, story_names):
    print(f"Processing {run_dir.name}...")

    with open(run_dir / "recall_matrices_model.pkl", "rb") as f:
        matrices_model = pickle.load(f)

    story_recall_segments = load_story_recall_segments_for_testset(testset, story_names)
    human_matrices = load_human_matrices_for_testset(
        testset, story_names, story_recall_segments
    )

    # filter out all-zero comparison matrices (same as eval loop)
    filtered = [
        (story_name, sub_id, story_segs, recall_segs)
        for story_name, sub_id, story_segs, recall_segs in story_recall_segments
        if not (human_matrices.get((story_name, sub_id), np.zeros((1, 1))) == 0).all()
        and (story_name, sub_id) in human_matrices
    ]

    assert len(filtered) == len(matrices_model), (
        f"Mismatch: {len(filtered)} subjects vs {len(matrices_model)} model matrices"
    )

    output_path = run_dir / "inspection.txt"
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, (story_name, sub_id, story_segs, recall_segs) in enumerate(filtered):
            rm_model = matrices_model[idx]
            rm_human = (human_matrices[(story_name, sub_id)] > 0).astype(float)

            diff = np.abs(rm_human - rm_model)
            disagreed_indices = np.where(diff.sum(axis=0) > 0)[0].tolist()

            out.write("=" * 80 + "\n")
            out.write(f"STORY: {story_name} | SUBJECT: {sub_id}\n")
            out.write(
                f"Disagreements: {len(disagreed_indices)}"
                f" / {len(recall_segs)} recall segments\n"
            )
            out.write("=" * 80 + "\n\n")

            if not disagreed_indices:
                out.write("No disagreements.\n\n")
                continue

            for recall_idx in disagreed_indices:
                out.write(f"  RECALL SEGMENT {recall_idx}\n")
                out.write(f"  {'-' * 76}\n")
                out.write(f"  Text: {recall_segs[recall_idx].strip()}\n\n")

                model_matches = np.where(rm_model[:, recall_idx] > 0)[0].tolist()
                human_matches = np.where(rm_human[:, recall_idx] > 0)[0].tolist()

                false_positives = [i for i in model_matches if i not in human_matches]
                false_negatives = [i for i in human_matches if i not in model_matches]

                if false_positives:
                    out.write("  FALSE POSITIVES (model said yes, human said no):\n")
                    for i in false_positives:
                        out.write(f"    [{i}] {story_segs[i].strip()}\n")
                    out.write("\n")

                if false_negatives:
                    out.write("  FALSE NEGATIVES (model said no, human said yes):\n")
                    for i in false_negatives:
                        out.write(f"    [{i}] {story_segs[i].strip()}\n")
                    out.write("\n")

    print(f"  Written to {output_path}")


if __name__ == "__main__":
    for run in RUNS:
        write_inspection(
            run_dir=run["run_dir"],
            testset=run["testset"],
            story_names=run["story_names"],
        )
