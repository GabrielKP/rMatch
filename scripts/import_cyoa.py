from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def import_cyoa(cyoa_path: Path | str):
    cyoa_path = Path(cyoa_path)
    if not cyoa_path.exists():
        raise FileNotFoundError(f"Cyoa path {cyoa_path} does not exist")

    for base_story in ["alice", "monthiversary"]:
        # 1. recall data
        recall_dir = cyoa_path / base_story / "3_pasv"
        recall_paths = sorted(list(recall_dir.glob("*recall.xlsx")))

        for recall_path in recall_paths:
            # filename/storyname
            filestem_splits = recall_path.stem.split("_")
            story_version = filestem_splits[0][-(len(filestem_splits[0]) - 2) :]
            story_name = f"{base_story}_{story_version}"
            sub_id = filestem_splits[1]
            assert filestem_splits[2].startswith("rate-recall")

            output_dir_recalls_segmented = (
                Path("data") / "cyoa" / story_name / "recalls" / "segmentation"
            )
            output_dir_recalls_segmented.mkdir(parents=True, exist_ok=True)

            recall_df = pd.read_excel(recall_path)

            recall_df = recall_df.rename(
                columns={
                    "recalled_events": "events",
                    "recall_in_temporal_order": "text",
                }
            )
            recall_df["segment"] = list(range(1, len(recall_df) + 1))
            recall_df = recall_df[["segment", "events", "text"]]

            recall_df.to_csv(
                output_dir_recalls_segmented / f"{sub_id}.csv",
                index=False,
            )

        # 2. story transcript data
        transcript_paths = sorted(list(recall_dir.glob("*events.xlsx")))
        for transcript_path in transcript_paths:
            if transcript_path.stem.startswith("~$"):
                continue
            # filename/storyname
            filestem_splits = transcript_path.stem.split("_")
            story_version = filestem_splits[0][-(len(filestem_splits[0]) - 2) :]
            story_name = f"{base_story}_{story_version}"
            assert filestem_splits[2].startswith("events")

            # output dir
            output_dir_transcripts = Path("data") / "cyoa" / story_name / "transcripts"
            output_dir_transcripts.mkdir(parents=True, exist_ok=True)

            # load data
            transcript_df = pd.read_excel(transcript_path)
            transcript_df = transcript_df.rename(columns={"story_texts": "text"})

            transcript_df = transcript_df[["event", "text"]]

            transcript_df.to_csv(
                output_dir_transcripts / f"{story_name}.csv",
                index=False,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cyoa-path", type=str, default="downloads/cyoa")
    args = parser.parse_args()

    cyoa_path = args.cyoa_path

    import_cyoa(cyoa_path=cyoa_path)
