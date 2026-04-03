import pickle
from collections import defaultdict
from pathlib import Path

from rmatch.load import load_ratings_dict, load_story_recall_segments
from rmatch.utils import ratings_single_sub_to_matrix

STORY_NAMES_MEMSEARCH10 = [
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

output_dir = Path("data/eval/fixed_matrices")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading human ratings...")
ratings_dicts_memsearch = defaultdict(dict)
for story_name in STORY_NAMES_MEMSEARCH10:
    ratings_dict = load_ratings_dict(
        story_name=story_name,
        matcher_name="human",
        story_segmentation_method="seg_c",
        recall_segmentation_method="sentences",
    )
    n_story_segments = ratings_dict["n_story_segments"]
    for sub_id, single_sub_ratings in ratings_dict["ratings"].items():
        ratings_dicts_memsearch[story_name][sub_id] = ratings_single_sub_to_matrix(
            single_sub_ratings, n_story_segments
        )

print("Building story recall segments list...")
story_recall_segments_memsearch = []
for story_name in STORY_NAMES_MEMSEARCH10:
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

print("Reconstructing comparison matrices...")
recall_matrices_comparison = []
for (
    story_name,
    sub_id,
    story_segments,
    recall_segments,
) in story_recall_segments_memsearch:
    rm_comparison = ratings_dicts_memsearch[story_name][sub_id]
    if (rm_comparison == 0).all():
        print(f"  Skipping {story_name=} {sub_id=}: all zero")
        continue
    recall_matrices_comparison.append(rm_comparison)

output_path = output_dir / "recall_matrices_memsearch10.pkl"
with open(output_path, "wb") as f:
    pickle.dump(recall_matrices_comparison, f)
print(f"Saved {len(recall_matrices_comparison)} matrices to {output_path}")
