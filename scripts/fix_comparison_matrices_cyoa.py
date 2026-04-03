import pickle
from pathlib import Path

from rmatch.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
)

TESTSETS = {
    "cyoa_alice10": [
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
}

output_dir = Path("data/eval/fixed_matrices")
output_dir.mkdir(parents=True, exist_ok=True)

for testset, story_names in TESTSETS.items():
    print(f"Reconstructing {testset}...")
    story_recall_segments = load_cyoa_story_recall_segments(story_names=story_names)

    recall_matrices_comparison = []
    for story_name, sub_id, story_segments, recall_segments in story_recall_segments:
        rm_comparison = load_cyoa_recall_matrix_human_binary(
            story_name=story_name, sub_id=sub_id
        )
        if (rm_comparison == 0).all():
            print(f"  Skipping {story_name=} {sub_id=}: all zero")
            continue
        recall_matrices_comparison.append(rm_comparison)

    output_path = output_dir / f"recall_matrices_{testset}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(recall_matrices_comparison, f)
    print(f"  Saved {len(recall_matrices_comparison)} matrices to {output_path}")
