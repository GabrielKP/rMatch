from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_recalls(dataset: str, n_subjects: int) -> pd.DataFrame:
    if dataset == "filmfest":
        recall_dir = Path("data") / "filmfest" / "recalls"
        recall_paths = sorted(list(recall_dir.glob("*.csv")))
        recall_dfs: list[pd.DataFrame] = list()
        for recall_path in recall_paths:
            recall_df = pd.read_csv(recall_path)
            recall_df["sub_id"] = recall_path.stem.split("-")[1]
            recall_dfs.append(recall_df)
        return pd.concat(recall_dfs[:n_subjects])
    else:
        raise ValueError(f"Dataset {dataset} not found")


def load_transcripts(dataset: str, rater: str = "JL") -> pd.DataFrame:
    if dataset == "filmfest":
        return pd.read_csv(
            Path("data") / "filmfest" / "transcripts" / f"{rater}_transcript.csv"
        )
    else:
        raise ValueError(f"Dataset {dataset} not found")


def get_rm_filmfest(
    recall_df: pd.DataFrame, transcript_df: pd.DataFrame
) -> tuple[list[str], list[np.ndarray]]:
    """Returns the recall matrices for each filmfest subject.

    Recall matrix: rows: story segments, columns: recall segments
    """

    M = transcript_df["seg_num"].max().item()
    # 1. convert multi-matched segments into multiple rows
    rms: list[np.ndarray] = list()
    sub_ids: list[str] = list()
    for sub_id, sub_recall_df in recall_df.groupby("sub_id"):
        rm_rows: list[np.ndarray] = list()
        for _, row in sub_recall_df.iterrows():
            rm_row = np.zeros(M)
            add_row = False
            for seg_num_str in row["scene"].split(","):  # type: ignore
                seg_num = int(seg_num_str)
                # skip if seg_num is not a scene label:
                # https://github.com/jchenlab-jhu/filmfest/blob/main/recall_scenematched/_scenematching_codingscheme.xlsx
                if seg_num > M or seg_num < 1:
                    continue

                add_row = True
                rm_row[seg_num - 1] = 1
            if add_row:
                rm_rows.append(rm_row)
        rm = np.array(rm_rows)
        rms.append(rm.T)
        sub_ids.append(sub_id)  # type: ignore

    return sub_ids, rms


def get_color_matrix(story_indices: np.ndarray, rm: np.ndarray) -> np.ndarray:
    color_matrix = np.copy(rm)
    for idx in range(1, story_indices.max()):
        color_matrix[story_indices == idx] = idx
    return color_matrix


def plot_rm(
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
    recall_df = load_recalls(dataset="filmfest", n_subjects=3)
    transcript_df = load_transcripts(dataset="filmfest")
    story_indices = transcript_df.groupby("seg_num")["movie_num"].first().to_numpy()

    sub_ids, rms = get_rm_filmfest(recall_df, transcript_df)
    plot_rm(story_indices=story_indices, sub_titles=sub_ids, rms=rms)
