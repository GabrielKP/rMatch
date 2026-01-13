from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

from recall_matrix import get_logger

log = get_logger(__name__)


def load_cyoa_story_recall_segments(
    story_names: list[str] | None = None,
) -> list[tuple[str, str, list[str], list[str]]]:
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

    if story_names is None:
        story_names = sorted(list([s.stem for s in cyoa_dir.glob("*")]))

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


def load_filmfest_story_recall_segments(
    story_names: list[str],
    sub_ids: list[str] | None = None,
    transcript_rater: str = "JL",
) -> list[tuple[str, str, list[str], list[str]]]:
    """Returns the story segments and recall segments for the given story names.

    Parameters
    ----------
    story_names: list[str]
        list of story names (movie names)
    sub_ids: list[str] | None
        list of subject ids included
    transcript_rater: "JL" | "KM" | "RC"
        rater of the transcripts

    Returns
    -------
    story_recall_segments: list[tuple[str, str, list[str], list[str]]]
        - story_name: name of the story
        - sub_id: subject id
        - story_segments: list of story segments
        - recall_segments: list of recall segments
    """

    story_recall_segments: list[tuple[str, str, list[str], list[str]]] = list()

    # preload transcripts
    transcript_segments_dict: dict[str, list[str]] = dict()
    all_transcripts = pd.read_csv(
        Path("data") / "filmfest" / "transcripts" / f"{transcript_rater}_transcript.csv"
    )
    for story_name in story_names:
        transcript_df = all_transcripts.loc[all_transcripts["movie_name"] == story_name]
        transcript_segments_dict[story_name] = transcript_df["text"].tolist()

    # iterate over participants
    filmfest_recall_dir = Path("data") / "filmfest" / "recalls"
    recall_paths = sorted(list(filmfest_recall_dir.glob("sub-*.csv")))
    for recall_path in recall_paths:
        sub_id = recall_path.stem
        if sub_ids is not None and sub_id not in sub_ids:
            continue
        all_recall_df = pd.read_csv(recall_path)

        for story_name in story_names:
            recall_df = all_recall_df.loc[all_recall_df["movie_name"] == story_name]
            recall_segments = recall_df["text"].tolist()
            story_segments = transcript_segments_dict[story_name]
            story_recall_segments.append(
                (story_name, sub_id, story_segments, recall_segments)
            )

    if len(story_recall_segments) == 0:
        raise ValueError(f"No recalls found for {story_names=}, {sub_ids=}")

    return story_recall_segments


def load_story_recall_segments(
    story_name: str,
    story_segment_method: str | None = None,
    recall_segment_method: str | None = None,
    sub_ids: list[str] | None = None,
) -> list[tuple[str, list[str], list[str]]]:
    """Returns the story segments and recall segments for the given story name.

    Parameters
    ----------
    story_name: str | Path
    story_segment_method: str | None
    sub_ids: list[str] | None

    Returns
    -------
    story_recall_segments: list[tuple[str, list[str], list[str]]]
        - sub_id: subject id
        - story_segments: list of story segments
        - recall_segments: list of recall segments
    """
    story_path = Path("data") / "stories-and-recalls" / story_name
    if not story_path.exists():
        raise FileNotFoundError(
            f"Story dir {story_path} for {story_name=} does not exist"
        )

    # try to auto-select story segment method
    if story_segment_method is None:
        # get all options
        story_segment_methods_available = sorted(
            [ssm.stem for ssm in (story_path / "transcripts").glob("*")]
        )
        assert len(story_segment_methods_available) > 0, (
            f"No story data found: {story_path / 'transcripts'}"
        )
        if len(story_segment_methods_available) == 1:
            story_segment_method = story_segment_methods_available[0]
        elif "sentences" in story_segment_methods_available:
            story_segment_method = "sentences"
        else:
            raise ValueError(
                f"Choose --story_segment_method: {story_segment_methods_available}"
            )

    # load story segments
    story_segments = story_path / "transcripts" / f"{story_segment_method}.txt"
    story_segments = story_segments.read_text().split("\n")

    # try to auto-select recall segment method
    if recall_segment_method is None:
        # get all options
        recall_segment_methods_available = sorted(
            [ssm.stem for ssm in (story_path / "recalls").glob("*")]
        )
        assert len(recall_segment_methods_available) > 0, (
            f"No recall data found: {story_path / 'recalls'}"
        )
        if len(recall_segment_methods_available) == 1:
            recall_segment_method = recall_segment_methods_available[0]
        elif "sentences" in recall_segment_methods_available:
            recall_segment_method = "sentences"
        else:
            raise ValueError(
                f"Choose --recall_segment_method: {recall_segment_methods_available}"
            )

    # load recall segments
    recall_paths = (story_path / "recalls" / recall_segment_method).glob("*.txt")
    story_recall_segments: list[tuple[str, list[str], list[str]]] = list()
    for recall_path in recall_paths:
        sub_id = recall_path.stem
        if sub_ids is not None and sub_id not in sub_ids:
            continue
        recall_segments = recall_path.read_text().split("\n")
        story_recall_segments.append((sub_id, story_segments, recall_segments))

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

        events_str = row_recall["events"]
        if not isinstance(events_str, str) and np.isnan(events_str):
            continue
        if isinstance(events_str, Number):
            try:
                temp = int(events_str)  # type: ignore
                events_str = str(temp)
            except ValueError:
                continue

        events_str = events_str.replace(".", ",")  # type: ignore
        for idx_event_str in events_str.split(","):  # type: ignore
            idx_event = int(idx_event_str) - 1

            if idx_event > 99999:
                # some ratings are merged numbers: e.g. 123163 -> 123, 163
                # have extra inner loop to split numbers
                if not (len(idx_event_str) % 3 == 0):
                    print(f"Invalid event index: {sub_id=} {story_name=} {idx_event=}")
                    continue
                for i in range(len(idx_event_str) // 3):
                    idx_event = int(idx_event_str[i * 3 : (i + 1) * 3]) - 1
                    assert (
                        idx_event == int(transcript_df.loc[idx_event, "event"]) - 1
                    ), "Event data not in order."
                    recall_matrix[idx_event, idx_recall] = 1  # type: ignore
            elif idx_event > len(transcript_df):
                # TODO handle this properly
                print(f"Invalid event index: {sub_id=} {story_name=} {idx_event=}")
                continue
            else:
                # sanity check
                assert idx_event == int(transcript_df.loc[idx_event, "event"]) - 1, (
                    "Event data not in order."
                )
                recall_matrix[idx_event, idx_recall] = 1  # type: ignore
    return recall_matrix


def load_filmfest_recall_matrix_human_binary(
    story_name: str, sub_id: str
) -> np.ndarray:
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

    # get recall df
    recall_path = Path("data") / "filmfest" / "recalls" / f"{sub_id}.csv"
    all_recall_df = pd.read_csv(recall_path)
    recall_df = all_recall_df.loc[all_recall_df["movie_name"] == story_name]

    # get transcript df
    transcript_path = Path("data") / "filmfest" / "transcripts" / "JL_transcript.csv"
    # trancript rater does not matter as event segmentation is the same for all raters
    all_transcript_df = pd.read_csv(transcript_path)
    transcript_df = all_transcript_df.loc[all_transcript_df["movie_name"] == story_name]

    max_seg_num = all_transcript_df["seg_num"].max()

    # get movie seg_nums to match recall seg_nums
    movie_seg_nums: list[int] = transcript_df["seg_num"].unique().tolist()

    # to make things easier, treat recall as rows and transpose later
    rm_rows: list[np.ndarray] = list()
    M: int = len(movie_seg_nums)
    for _, row in recall_df.iterrows():
        rm_row = np.zeros(M)
        for seg_num_str in row["scene"].split(","):  # type: ignore
            seg_num = int(seg_num_str)
            # if seg_num is not a scene label:
            # do not set any row index is set to 1
            # https://github.com/jchenlab-jhu/filmfest/blob/main/recall_scenematched/_scenematching_codingscheme.xlsx
            if seg_num > max_seg_num or seg_num < 1:
                continue
            try:
                idx_row = movie_seg_nums.index(seg_num)
                rm_row[idx_row] = 1
            except ValueError as err:
                print(f"{seg_num=} for recall {row=} not found in movie {story_name=}")
                raise err
        rm_rows.append(rm_row)
    recall_matrix = np.array(rm_rows).T

    return recall_matrix


def load_nfrd_recall_matrix_human_mi(
    story_name: str, sub_id: str, rater: str
) -> np.ndarray:
    """Returns the recall matrix for the given story name and subject id.

    Parameters
    ----------
    story_name: str
        name of the story
    sub_id: str
        subject id
    rater: str
        who rated the file, all lowercase

    Returns
    -------
    recall_matrix: np.ndarray
        recall matrix of shape (len(story_segments), len(recall_segments))
    """

    base = Path(__file__).resolve().parents[2]
    ratings_path = (
        base
        / "data"
        / "nfrd"
        / "recalls"
        / "human_mi"
        / story_name
        / f"{sub_id}_{story_name}_{rater}.csv"
    )

    df = pd.read_csv(ratings_path, dtype={"mi": int})
    recall_segs = df["recall_segment"].unique().tolist()
    story_segs = df["story_segment"].unique().tolist()

    recall_matrix = np.full((len(story_segs), len(recall_segs)), 0)
    for r_idx, r_seg in enumerate(recall_segs):
        for s_idx, s_seg in enumerate(story_segs):
            pair_mi = df.loc[
                (df["recall_segment"] == r_seg) & (df["story_segment"] == s_seg), "mi"
            ]
            recall_matrix[s_idx, r_idx] = int(pair_mi.iat[0])
    recall_matrix = recall_matrix / 100

    return recall_matrix
