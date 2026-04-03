"""Create the rMatch benchmark from existing CYOA and MemSearch data.

Usage:
    uv run scripts/create_benchmark.py
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

BENCHMARK_DIR = Path("benchmark")

DATASETS = {
    "alice": {
        "stories": [
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
        "modality": "text",
        "source": "cyoa",
    },
    "monthiversary": {
        "stories": [
            "monthiversary_3",
            "monthiversary_4",
            "monthiversary_10",
            "monthiversary_14",
            "monthiversary_19",
            "monthiversary_23",
            "monthiversary_25",
        ],
        "modality": "text",
        "source": "cyoa",
    },
    "memsearch": {
        "stories": [
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
        "modality": "audiovisual",
        "source": "memsearch",
    },
}

# seg_b -> events, seg_c -> scenes
MEMSEARCH_SEG_RENAME = {"seg_b": "events", "seg_c": "scenes"}


def write_json(path: Path, data: dict, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"  wrote {path}")


def parse_cyoa_events(
    events_raw, sub_id: str, story_name: str, n_events: int
) -> list[int]:
    """Parse the events column from a CYOA recall CSV into 0-based story indices.

    Mirrors the logic in load.load_cyoa_recall_matrix_human_binary.
    """
    if isinstance(events_raw, float) and math.isnan(events_raw):
        return []
    if isinstance(events_raw, (int, float)):
        try:
            events_raw = str(int(events_raw))
        except ValueError:
            return []

    events_raw = str(events_raw).replace(".", ",")
    indices = []
    for idx_event_str in events_raw.split(","):
        idx_event_str = idx_event_str.strip()
        if not idx_event_str:
            continue
        idx_event = int(idx_event_str) - 1

        if idx_event > 99999:
            if len(idx_event_str) % 3 != 0:
                print(
                    f"  Warning: invalid merged event index:"
                    f" {sub_id=} {story_name=} {idx_event_str=}"
                )
                continue
            for i in range(len(idx_event_str) // 3):
                chunk_idx = int(idx_event_str[i * 3 : (i + 1) * 3]) - 1
                if 0 <= chunk_idx < n_events:
                    indices.append(chunk_idx)
                else:
                    print(
                        "  Warning: event index out of range:"
                        f" {sub_id=} {story_name=} idx={chunk_idx}"
                    )
        elif idx_event < 0 or idx_event >= n_events:
            print(
                f"  Warning: event index out of range:"
                f" {sub_id=} {story_name=} idx={idx_event}"
            )
        else:
            indices.append(idx_event)

    return sorted(set(indices))


def convert_cyoa_story(benchmark_dir: Path, story_name: str, dataset_name: str) -> None:
    """Convert a single CYOA story (alice or monthiversary) to benchmark format."""
    cyoa_dir = Path("data") / "cyoa" / story_name
    out_dir = benchmark_dir / dataset_name / story_name

    # 1. Transcript
    transcript_path = cyoa_dir / "transcripts" / f"{story_name}.csv"
    transcript_df = pd.read_csv(transcript_path)
    story_segments = transcript_df["text"].tolist()
    n_events = len(story_segments)

    write_json(
        out_dir / "transcripts" / "scenes.json",
        {
            "type": "transcript",
            "segmentation_method": "scenes",
            "segments": story_segments,
        },
    )

    # 2 & 3. Recalls + matches (read recall CSVs once for both)
    recall_dir = cyoa_dir / "recalls" / "segmentation"
    recall_paths = sorted(recall_dir.glob("*.csv"))

    all_recalls: dict[str, list[str]] = {}
    all_ratings: dict[str, list] = {}

    for recall_path in recall_paths:
        sub_id = recall_path.stem
        recall_df = pd.read_csv(recall_path)
        recall_segments = recall_df["text"].tolist()
        all_recalls[sub_id] = recall_segments

        sub_ratings = []
        for idx_recall, row in recall_df.iterrows():
            matched_story_indices = parse_cyoa_events(
                row["events"], sub_id, story_name, n_events
            )
            sub_ratings.append([int(idx_recall), matched_story_indices])  # type: ignore
        all_ratings[sub_id] = sub_ratings

    write_json(
        out_dir / "recalls" / "sub_sentences.json",
        {
            "type": "recalls",
            "segmentation_method": "sub_sentences",
            "recalls": all_recalls,
        },
    )

    write_json(
        out_dir / "matches" / "human-scenes-sub_sentences.json",
        {
            "type": "matches",
            "rater": "human",
            "story_segmentation": "scenes",
            "recall_segmentation": "sub_sentences",
            "n_story_segments": n_events,
            "ratings": all_ratings,
        },
    )


def convert_memsearch_story(benchmark_dir: Path, story_name: str) -> None:
    """Convert a single MemSearch story to benchmark format."""
    src_dir = Path("data") / "stories-and-recalls" / story_name
    out_dir = benchmark_dir / "memsearch" / story_name

    # 1. Transcripts (seg_b -> events, seg_c -> scenes)
    for src_name, dst_name in MEMSEARCH_SEG_RENAME.items():
        transcript_path = src_dir / "transcripts" / f"{src_name}.txt"
        segments = [
            seg for seg in transcript_path.read_text().split("\n") if seg.strip()
        ]
        write_json(
            out_dir / "transcripts" / f"{dst_name}.json",
            {
                "type": "transcript",
                "segmentation_method": dst_name,
                "segments": segments,
            },
        )

    # 2. Recalls
    recall_dir = src_dir / "recalls" / "sentences"
    recall_paths = sorted(recall_dir.glob("*.txt"))
    all_recalls: dict[str, list[str]] = {}
    for recall_path in recall_paths:
        sub_id = recall_path.stem
        segments = [seg for seg in recall_path.read_text().split("\n") if seg.strip()]
        all_recalls[sub_id] = segments

    write_json(
        out_dir / "recalls" / "sentences.json",
        {
            "type": "recalls",
            "segmentation_method": "sentences",
            "recalls": all_recalls,
        },
    )

    # 3. Matches (one per segmentation method)
    for src_name, dst_name in MEMSEARCH_SEG_RENAME.items():
        ratings_path = src_dir / "ratings" / f"human-ssm_{src_name}-rsm_sentences.json"
        with open(ratings_path) as f:
            src_ratings = json.load(f)

        write_json(
            out_dir / "matches" / f"human-{dst_name}-sentences.json",
            {
                "type": "matches",
                "rater": "human",
                "story_segmentation": dst_name,
                "recall_segmentation": "sentences",
                "n_story_segments": src_ratings["n_story_segments"],
                "ratings": src_ratings["ratings"],
            },
        )


def create_benchmark(benchmark_dir: Path) -> None:
    if benchmark_dir.exists():
        # Preserve hand-edited files like README.md
        readme = benchmark_dir / "README.md"
        readme_backup = readme.read_text() if readme.exists() else None
        shutil.rmtree(benchmark_dir)
        print(f"Removed existing {benchmark_dir}/")
        if readme_backup is not None:
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            readme.write_text(readme_backup)
            print(f"  preserved {readme}")

    # Top-level description
    write_json(
        benchmark_dir / "description.json",
        {
            "name": "recall matching benchmark",
            "version": "1.0",
            "description": "",
            "datasets": list(DATASETS.keys()),
        },
        indent=2,
    )

    for dataset_name, dataset_info in DATASETS.items():
        print(f"\n=== {dataset_name} ===")
        stories = dataset_info["stories"]

        # Dataset metadata
        write_json(
            benchmark_dir / dataset_name / "dataset.json",
            {
                "name": dataset_name,
                "description": "",
                "stories": stories,
                "num_stories": len(stories),
                "modality": dataset_info["modality"],
            },
            indent=2,
        )

        for story_name in stories:
            print(f"\n  --- {story_name} ---")
            if dataset_info["source"] == "cyoa":
                convert_cyoa_story(benchmark_dir, story_name, dataset_name)
            else:
                convert_memsearch_story(benchmark_dir, story_name)

    print("\nDone.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dir", type=str, default=None)
    args = args.parse_args()

    create_benchmark(BENCHMARK_DIR if args.dir is None else Path(args.dir))
