import json
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

from rmatch import get_logger

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
    transcript_matcher: str = "JL",
) -> list[tuple[str, str, list[str], list[str]]]:
    """Returns the story segments and recall segments for the given story names.

    Parameters
    ----------
    story_names: list[str]
        list of story names (movie names)
    sub_ids: list[str] | None
        list of subject ids included
    transcript_matcher: "JL" | "KM" | "RC"
        matcher of the transcripts

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
        Path("data")
        / "filmfest"
        / "transcripts"
        / f"{transcript_matcher}_transcript.csv"
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


def load_recall_segments(
    story_name: str,
    method: str | None = None,
    sub_ids: list[str] | None = None,
) -> tuple[list[tuple[str, list[str]]], str]:
    """Returns the recall segments for the given story name and method."""
    story_path = Path("data") / "stories-and-recalls" / story_name
    if not story_path.exists():
        raise FileNotFoundError(
            f"Story dir {story_path} for {story_name=} does not exist"
        )

    # try to auto-select recall segment method
    if method is None:
        # get all options
        methods_available = sorted(
            [ssm.stem for ssm in (story_path / "recalls").glob("*")]
        )
        assert len(methods_available) > 0, (
            f"No recall data found: {story_path / 'recalls'}"
        )
        if len(methods_available) == 1:
            method = methods_available[0]
        elif "sentences" in methods_available:
            method = "sentences"
        else:
            raise ValueError(f"Choose --method: {methods_available}")
        log.info(f"Auto-selected recall segmentation method: {method}")

    # load recall segments
    recall_paths = (story_path / "recalls" / method).glob("*.txt")
    recall_segments_list: list[tuple[str, list[str]]] = list()
    for recall_path in recall_paths:
        sub_id = recall_path.stem
        if sub_ids is not None and sub_id not in sub_ids:
            continue
        recall_segments = [
            seg for seg in recall_path.read_text().split("\n") if seg.strip()
        ]
        recall_segments_list.append((sub_id, recall_segments))

    return recall_segments_list, method
