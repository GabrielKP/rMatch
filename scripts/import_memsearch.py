"""Import memsearch data."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

from recall_matrix.utils import get_param_str

console = Console()

MEMSEARCH_NUM_TO_MOVIE = {
    1: "Breadland",
    2: "Ednora",
    3: "From-Dad-to-Son",
    4: "Heartstrings",
    5: "Hollow",
    6: "I-Love-Death",
    7: "Ichthys",
    8: "Laundry",
    9: "Mismatched",
    10: "Mop",
    11: "My-Cat-Lucy",
    12: "Numb",
    13: "Queen-of-Basketball",
    14: "Stapler",
    15: "Synesthesia",
    16: "The-Docks",
    17: "The-Gift",
    18: "The-Port",
    19: "The-Soup",
    20: "Thief",
}

# explicitivity > automaticity
MOVIE_NAME_TO_STORY_NAME = {
    "Breadland": "breadland",
    "Ednora": "ednora",
    "From-Dad-to-Son": "from_dad_to_son",
    "Heartstrings": "heartstrings",
    "Hollow": "hollow",
    "I-Love-Death": "i_love_death",
    "Ichthys": "ichthys",
    "Laundry": "laundry",
    "Mismatched": "mismatched",
    "Mop": "mop",
    "My-Cat-Lucy": "my_cat_lucy",
    "Numb": "numb",
    "Queen-of-Basketball": "queen_of_basketball",
    "Stapler": "stapler",
    "Synesthesia": "synesthesia",
    "The-Docks": "the_docks",
    "The-Gift": "the_gift",
    "The-Port": "the_port",
    "The-Soup": "the_soup",
    "Thief": "thief",
}


def _seg_match_to_str(val):
    """Convert SEG-* Match cell to string. If Excel already converted e.g. '3,4' to a
    date (March 4), recover segment indices as 'month,day'."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if pd.isna(val):
        return ""
    if isinstance(val, (pd.Timestamp, datetime)):
        return f"{val.month},{val.day}"
    return str(val)


def import_memsearch(memsearch_path: Path | str):
    memsearch_path = Path(memsearch_path)
    if not memsearch_path.exists():
        raise FileNotFoundError(f"Memsearch path {memsearch_path} does not exist")

    transcript_dir = memsearch_path / "Trimmed Movies Annotations"
    recall_dir = memsearch_path / "Completed Scene-Matched Files"

    for movie_num, movie_name in MEMSEARCH_NUM_TO_MOVIE.items():
        story_name = MOVIE_NAME_TO_STORY_NAME[movie_name]

        # 1. import transcripts
        transcript_df = pd.read_excel(transcript_dir / f"{movie_name}_annotations.xlsx")
        story_name = MOVIE_NAME_TO_STORY_NAME[movie_name]
        transcript_segc_path = (
            Path("data")
            / "stories-and-recalls"
            / story_name
            / "transcripts"
            / "seg_c.txt"
        )
        transcript_segc_path.parent.mkdir(parents=True, exist_ok=True)
        segc_segments = transcript_df["SEG-C Description"].tolist()
        transcript_segc_path.write_text("\n".join(segc_segments) + "\n")
        transcript_segb_path = (
            Path("data")
            / "stories-and-recalls"
            / story_name
            / "transcripts"
            / "seg_b.txt"
        )
        segb_segments = [
            x.strip()
            for x in transcript_df["SEG-B Description"].tolist()
            if isinstance(x, str)
        ]
        transcript_segb_path.write_text("\n".join(segb_segments) + "\n")

        console.print(f" > ({story_name}) transcripts imported", style="yellow")
        print(f"   > SEG-C: {transcript_segc_path}")
        print(f"   > SEG-B: {transcript_segb_path}")

        # make a file to mark this as audiovisual
        (
            Path("data") / "stories-and-recalls" / story_name / ".is_audiovisual"
        ).write_text("")

        # 2 & 3 import recall segmentation & ratings
        output_dict_segc = {
            "rater_name": "human",
            "story_name": story_name,
            "story_segmentation_method": "seg_c",
            "recall_segmentation_method": "sentences",
            "n_story_segments": len(segc_segments),
            "output_scores": False,
        }
        output_dict_segb = {
            "rater_name": "human",
            "story_name": story_name,
            "story_segmentation_method": "seg_b",
            "recall_segmentation_method": "sentences",
            "n_story_segments": len(segb_segments),
            "output_scores": False,
        }

        ratings_segc = dict()
        ratings_segb = dict()
        for sub_num in range(1, 41):
            sub_id = f"sub-{sub_num:03d}"
            sub_recall_paths = list(
                recall_dir.glob(f"memsrc{sub_num:02d}_recall_annotation*.xlsx")
            )
            if len(sub_recall_paths) == 0:
                print(f"    ({story_name}) ({sub_id}) recall data not found. Skipping.")
                continue
            if len(sub_recall_paths) > 1:
                print(
                    f"    ({story_name}) ({sub_id}) multiple recall data found:"
                    f" {sub_recall_paths}."
                )
                print("    Taking the first one")
            recall_path = sub_recall_paths[0]
            recall_df = pd.read_excel(
                recall_path,
                converters={
                    "SEG-C Match": _seg_match_to_str,
                    "SEG-B Match": _seg_match_to_str,
                },
            )

            if "Movie Label" not in recall_df.columns:
                print(
                    f"    ({story_name}) ({sub_id}) columns: {recall_df.columns.tolist()}"
                )
                continue

            movied_recall_df = recall_df[recall_df["Movie Label"] == movie_num]
            if len(movied_recall_df) == 0:
                print(
                    f"    ({story_name}) ({sub_id}) file found,"
                    " but no recall for story."
                )
                continue

            # 2. import recall segmentation
            sentence_col = "Sentence"
            if sub_id == "sub-018":
                sentence_col = "Sentence "
            elif sub_id == "sub-031":
                sentence_col = "sentence"
            recall_segments = movied_recall_df[sentence_col].tolist()
            recall_segments = [x.strip() for x in recall_segments]
            recall_segments_path = (
                Path("data")
                / "stories-and-recalls"
                / story_name
                / "recalls"
                / "sentences"
                / f"{sub_id}.txt"
            )
            recall_segments_path.parent.mkdir(parents=True, exist_ok=True)
            recall_segments_path.write_text("\n".join(recall_segments) + "\n")

            console.print(
                f" > ({story_name}) ({sub_id}) recall segmentation imported",
                style="yellow",
            )
            print(f"   > {recall_segments_path}")

            # 3. import recall ratings
            matches_segc = movied_recall_df["SEG-C Match"].tolist()
            ratings_segc[sub_id] = list()
            for idx_recall_segment, match_segc in enumerate(matches_segc):
                filtered_matches = list()
                if not isinstance(match_segc, str):
                    match_segc = ""
                for x in match_segc.split(","):
                    if x == "":
                        continue
                    try:
                        x_int = int(x)
                    except ValueError:
                        print(
                            f"    ({story_name}) ({sub_id}) ({idx_recall_segment})"
                            f" {x=} is not a valid integer."
                        )
                        continue
                    if x_int == 900:
                        continue
                    if x_int > len(segc_segments):
                        print(
                            f"    ({story_name}) ({sub_id}) ({idx_recall_segment})"
                            f" {x_int=} is out of range."
                        )
                        continue
                    if x_int < 1:
                        print(
                            f"    ({story_name}) ({sub_id}) ({idx_recall_segment})"
                            f" {x_int=} is less than 1."
                        )
                        continue
                    filtered_matches.append(int(x) - 1)
                ratings_segc[sub_id].append((idx_recall_segment, filtered_matches))

            matches_segb = movied_recall_df["SEG-B Match"].tolist()
            ratings_segb[sub_id] = list()
            for idx_recall_segment, match_segb in enumerate(matches_segb):
                filtered_matches = list()
                if not isinstance(match_segb, str):
                    match_segb = ""
                for x in match_segb.split(","):
                    if x == "":
                        continue
                    try:
                        x_int = int(x)
                    except ValueError:
                        print(
                            f"    ({story_name}) ({sub_id}) ({idx_recall_segment})"
                            f" {x=} is not a valid integer."
                        )
                        continue
                    if x_int == 900 or x_int == 999 or x_int == 800:
                        continue
                    if x_int > len(segb_segments):
                        print(
                            f"    ({story_name}) ({sub_id}) ({idx_recall_segment})"
                            f" {x_int=} is out of range."
                        )
                        continue
                    if x_int < 1:
                        print(
                            f"    ({story_name}) ({sub_id}) ({idx_recall_segment})"
                            f" {x_int=} is less than 1."
                        )
                        continue
                    filtered_matches.append(int(x) - 1)
                ratings_segb[sub_id].append((idx_recall_segment, filtered_matches))

        output_dict_segc["ratings"] = ratings_segc
        output_dict_segb["ratings"] = ratings_segb

        # save as json
        segc_param_str = get_param_str(output_dict_segc)
        segb_param_str = get_param_str(output_dict_segb)

        output_dict_segc_path = (
            Path("data")
            / "stories-and-recalls"
            / story_name
            / "ratings"
            / f"{segc_param_str}.json"
        )
        output_dict_segb_path = (
            Path("data")
            / "stories-and-recalls"
            / story_name
            / "ratings"
            / f"{segb_param_str}.json"
        )

        output_dict_segc_path.parent.mkdir(parents=True, exist_ok=True)
        output_dict_segb_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_dict_segc_path, "w") as f:
            f.write(json.dumps(output_dict_segc) + "\n")
        with open(output_dict_segb_path, "w") as f:
            f.write(json.dumps(output_dict_segb) + "\n")

        console.print(f" > [{story_name}] seg_c ratings imported", style="yellow")
        print(f"   > {output_dict_segc_path}")
        console.print(f" > [{story_name}] seg_b ratings imported", style="yellow")
        print(f"   > {output_dict_segb_path}")


if __name__ == "__main__":
    memsearch_path = Path("downloads/memsearch")
    import_memsearch(memsearch_path=memsearch_path)
