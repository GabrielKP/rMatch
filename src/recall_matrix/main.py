import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from recall_matrix import console


def load_recalls(
    dataset: str,
    sub_ids: list[str] | None = None,
    movie_nums: list[int] | None = None,
) -> pd.DataFrame:
    if dataset == "filmfest":
        recall_dir = Path("data") / "filmfest" / "recalls"
        recall_paths = sorted(list(recall_dir.glob("*.csv")))
        recall_dfs: list[pd.DataFrame] = list()
        for recall_path in recall_paths:
            recall_df = pd.read_csv(recall_path)
            # filter movies
            if movie_nums is not None:
                recall_df = recall_df.loc[recall_df["movie_num"].isin(movie_nums)]

            # filter sub_ids
            sub_id = recall_path.stem
            if sub_ids is None or sub_id in sub_ids:
                recall_df["sub_id"] = sub_id
                recall_dfs.append(recall_df)

        recalls_df = pd.concat(recall_dfs)
    else:
        raise ValueError(f"Dataset {dataset} not found")

    if len(recalls_df) == 0:
        raise ValueError(f"No recalls found for {dataset=}, {sub_ids=}, {movie_nums=}")

    return recalls_df


def load_transcripts(dataset: str, rater: str = "JL") -> pd.DataFrame:
    if dataset == "filmfest":
        return pd.read_csv(
            Path("data") / "filmfest" / "transcripts" / f"{rater}_transcript.csv"
        )
    else:
        raise ValueError(f"Dataset {dataset} not found")


def get_filmfest_rm_binary(
    recalls_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
) -> tuple[list[str], list[int], list[np.ndarray]]:
    """Returns the recall matrices for each filmfest subject.

    Recall matrix: rows: story segments, columns: recall segments
    """

    # 1. convert human binary ratings into recall matrices
    rms: list[np.ndarray] = list()
    sub_ids: list[str] = list()
    movie_nums: list[int] = list()

    max_seg_num = transcript_df["seg_num"].max()

    # Iterate over subs
    for sub_id, sub_df in recalls_df.groupby("sub_id"):
        # Iterate over movies
        for movie_num, recall_df in sub_df.groupby("movie_num"):
            # get movie seg_nums to match recall seg_nums
            movie_seg_nums: list[int] = (
                transcript_df[transcript_df["movie_num"] == movie_num]["seg_num"]
                .unique()  # type: ignore
                .tolist()
            )

            # For each sub, movie -> new rm
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
                        print(
                            f"{seg_num=} for recall {row=} not"
                            f" found in movie {movie_num=}"
                        )
                        raise err
                rm_rows.append(rm_row)
            rm = np.array(rm_rows)
            rms.append(rm.T)
            sub_ids.append(sub_id)  # type: ignore
            movie_nums.append(movie_num)  # type: ignore

    return sub_ids, movie_nums, rms


def get_color_matrix(story_indices: np.ndarray, rm: np.ndarray) -> np.ndarray:
    color_matrix = np.copy(rm)
    for idx in range(1, story_indices.max()):
        color_matrix[story_indices == idx] = idx
    return color_matrix


def plot_rms(
    story_indices: np.ndarray,
    sub_titles: list[str],
    rms: list[np.ndarray],
    recall_times: list[np.ndarray] | None = None,
    story_times: list[np.ndarray] | None = None,
):
    """Plot the recall matrix

    Parameters
    ----------
    sub_titles : list[str]
        Titles for each subplot
    rms : list[np.ndarray]
        Recall matrix (M x N) rows: story segments, columns: recall segments
    recall_times : np.ndarray | None
        Recall times
    story_times : np.ndarray | None
        Story times
    """

    assert len(sub_titles) == len(rms)

    fig, axes = plt.subplots(
        1,
        len(sub_titles),
        figsize=(7 * len(sub_titles), 10),
    )

    for idx, (ax, sub_title, rm) in enumerate(zip(axes, sub_titles, rms)):
        # to plot the trajectories with story colors, use recall matrix
        # as mask over "bands" of the colors of the story.
        color_matrix = get_color_matrix(story_indices, rm)
        ax.imshow(color_matrix, alpha=rm, cmap="Set1")
        ax.set_facecolor("black")
        ax.set_title(sub_title)
        ax.set_xlabel("Recall segments")
        ax.set_ylabel("Story segments")
        ax.set_aspect(1)

        if recall_times is not None:
            ax.set_xticks(np.arange(len(recall_times[idx])))
            ax.set_xticklabels(recall_times[idx])

        if story_times is not None:
            ax.set_yticks(np.arange(len(story_times[idx])))
            ax.set_yticklabels(story_times[idx])

    fig.tight_layout()
    plt.show()
    return fig, axes


if __name__ == "__main__":
    sub_ids = ["sub-01"]
    movie_nums = [1, 2]
    mutual_information_method = "with_history"
    mutual_information_normalize = False
    recalls_df = load_recalls(
        dataset="filmfest", sub_ids=sub_ids, movie_nums=movie_nums
    )
    transcript_df = load_transcripts(dataset="filmfest")

    sub_ids_binary, movie_nums_binary, rms_binary = get_filmfest_rm_binary(
        recalls_df, transcript_df
    )
