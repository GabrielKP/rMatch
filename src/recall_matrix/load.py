from pathlib import Path

import numpy as np
import pandas as pd


def load_cyoa_story_recall_segments(
    story_names: list[str],
) -> list[tuple[list[str], list[str]]]:
    """Returns the story segments and recall segments for the given story names.

    Parameters
    ----------
    story_names: list[str]
        list of story names

    Returns
    -------
    story_recall_segments: list[tuple[str, str, list[str], list[str]]]
        - story_name: name of the story
        - sub_id: subject id
        - story_segments: list of story segments
        - recall_segments: list of recall segments
    """

    story_recall_segments: list[tuple[str, str, list[str], list[str]]] = list()

    cyoa_dir = Path("data") / "cyoa"
    if not cyoa_dir.exists():
        raise FileNotFoundError("Download & import cyoa data first.")

    for story_name in story_names:
        transcript_path = cyoa_dir / story_name / "transcripts" / f"{story_name}.csv"
        transcript_df = pd.read_csv(transcript_path)
        story_segments = transcript_df["text"].tolist()

        recall_dir = cyoa_dir / story_name / "recalls" / "segmentation"
        recall_paths = sorted(list(recall_dir.glob("*.csv")))

        for recall_path in recall_paths:
            recall_df = pd.read_csv(recall_path)
            sub_id = recall_path.stem
            recall_segments = recall_df["text"].tolist()
            story_recall_segments.append(
                (story_name, sub_id, story_segments, recall_segments)
            )

    return story_recall_segments


def load_cyoa_recall_matrix_human_binary(story_name: str, sub_id: str) -> np.ndarray:
    """Returns the recall matrix for the given story name and subject id.

    Parameters
    ----------
    story_name: str
        name of the story
    sub_id: str
        subject id

    Returns
    -------
    recall_matrix: np.ndarray
        recall matrix of shape (len(story_segments), len(recall_segments))
    """

    transcript_path = (
        Path("data") / "cyoa" / story_name / "transcripts" / f"{story_name}.csv"
    )
    transcript_df = pd.read_csv(transcript_path)

    recall_path = (
        Path("data")
        / "cyoa"
        / story_name
        / "recalls"
        / "segmentation"
        / f"{sub_id}.csv"
    )
    recall_df = pd.read_csv(recall_path)

    recall_matrix = np.zeros((len(transcript_df), len(recall_df)))
    for idx_recall, row_recall in recall_df.iterrows():
        assert idx_recall == int(row_recall["segment"]) - 1, "Sanity check failed."
        if not isinstance(row_recall["events"], str) and np.isnan(row_recall["events"]):
            continue
        for idx_event_str in row_recall["events"].split(","):  # type: ignore
            idx_event = int(idx_event_str) - 1
            if idx_event > 99999:
                # some ratings are merged numbers: e.g. 123163 -> 123, 163
                # have extra inner loop to split numbers
                assert len(idx_event_str) % 3 == 0, "cannot disambiguate merged numbers"
                for i in range(len(idx_event_str) // 3):
                    idx_event = int(idx_event_str[i * 3 : (i + 1) * 3]) - 1
                    assert (
                        idx_event == int(transcript_df.loc[idx_event, "event"]) - 1
                    ), "Event data not in order."
                    recall_matrix[idx_event, idx_recall] = 1
                continue

            # sanity check
            assert idx_event == int(transcript_df.loc[idx_event, "event"]) - 1, (
                "Event data not in order."
            )
            recall_matrix[idx_event, idx_recall] = 1
    return recall_matrix
