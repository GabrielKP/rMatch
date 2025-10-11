"""Imports data from https://github.com/jchenlab-jhu/filmfest/tree/main"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def import_filmfest(filmfest_path: Path | str):
    filmfest_path = Path(filmfest_path)
    if not filmfest_path.exists():
        raise FileNotFoundError(f"Filmfest path {filmfest_path} does not exist")

    # 1. import recall data
    recall_dir = filmfest_path / "recall_scenematched"
    recall_paths = sorted(list(recall_dir.glob("sub-*_recall.xlsx")))

    output_dir = Path("data") / "filmfest" / "recalls"
    output_dir.mkdir(parents=True, exist_ok=True)

    for recall_path in recall_paths:
        recall_df = pd.read_excel(recall_path)

        # rename columns
        col_map = {"movies": "movie", "scenes": "scene"}
        recall_df = recall_df.rename(columns=col_map)

        recall_df.to_csv(
            output_dir / recall_path.stem.replace("_recall", ".csv"), index=False
        )

    # 2. import movie/scene data
    transcript_dir = filmfest_path / "annotations"
    transcript_paths = sorted(list(transcript_dir.glob("FilmFestival*.xlsx")))

    output_dir = Path("data") / "filmfest" / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    for transcript_path in transcript_paths:
        transcript_df = pd.read_excel(transcript_path)
        # get index loc for Run 2
        idx_row_run_2 = int(
            transcript_df[transcript_df["Scanning run"] == "Run 2"].index[0]  # type: ignore
        )
        # fill missing values
        transcript_df = transcript_df.ffill(axis=0)
        # get last segment number before Run 2
        idx_col_segment_number = transcript_df.columns.get_loc("Segment number")
        run1_last_segment_number = transcript_df.iloc[
            idx_row_run_2 - 1, idx_col_segment_number
        ]
        # add to Run 2
        transcript_df.iloc[idx_row_run_2:, idx_col_segment_number] += (
            run1_last_segment_number
        )
        # remove
        # add subseg number
        transcript_df["subseg_num"] = (
            transcript_df.groupby("Segment number").cumcount() + 1
        )
        # add movie number
        movie_title_map = {
            "1. Cartoon intro 1": 12,
            "2. Catch Me If You Can": 1,
            "3. The Record": 2,
            "4. The Boyfriend": 3,
            "5. The Shoe": 4,
            "6. Keith Reynolds": 5,
            "8. The Rock": 6,
            "9. The Prisoner": 7,
            "10. The Black Hole": 8,
            "11. Post-it Love": 9,
            "12. Bus Stop": 10,
            "7. Cartoon intro 2": 11,
        }
        transcript_df["movie_num"] = transcript_df["Movie title"].map(movie_title_map)  # type: ignore

        # rename columns
        col_map = {
            "Segment number": "seg_num",
            "Movie title": "movie_title",
            "Segment start time (min.sec)": "seg_start_time",
            "Fine-grained segment start time (min.sec)": "subseg_start_time",
            "Description": "text",
        }
        transcript_df = transcript_df.rename(columns=col_map)

        # keep columns we want
        transcript_df = transcript_df[
            [
                "movie_num",
                "seg_num",
                "seg_start_time",
                "subseg_num",
                "subseg_start_time",
                "text",
            ]
        ]

        # transform column types
        transcript_df["movie_num"] = transcript_df["movie_num"].astype(int)
        transcript_df["seg_num"] = transcript_df["seg_num"].astype(int)
        transcript_df["subseg_num"] = transcript_df["subseg_num"].astype(int)
        transcript_df["seg_start_time"] = transcript_df["seg_start_time"].astype(float)
        transcript_df["subseg_start_time"] = transcript_df["subseg_start_time"].astype(
            float
        )
        transcript_df.to_csv(
            output_dir / f"{transcript_path.stem.split('_')[-1]}_transcript.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filmfest-path", type=str, default="../filmfest")
    args = parser.parse_args()

    filmfest_path = args.filmfest_path

    import_filmfest(filmfest_path=filmfest_path)
