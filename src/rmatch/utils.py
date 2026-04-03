from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_param_str(output_dict: dict) -> str:
    """Get the param string from the output dict."""

    rater_name = output_dict["rater_name"]
    recall_segmentation_method = output_dict["recall_segmentation_method"]
    story_segmentation_method = output_dict["story_segmentation_method"]
    output_scores_str = "-ouput_scores" if output_dict["output_scores"] else ""

    param_str = (
        f"{rater_name}"
        f"-ssm_{story_segmentation_method}"
        f"-rsm_{recall_segmentation_method}"
        f"{output_scores_str}"
    )
    return param_str


def ratings_single_sub_to_matrix(
    ratings_single_sub: list[tuple[int, list[int]]], n_story_segments: int
) -> np.ndarray:
    """Convert the ratings to a recall matrix.

    Returns
    -------
    recall_matrix: np.ndarray
        recall matrix of shape (n_story_segments, n_recall_segments)
    """
    n_recall_segments = len(ratings_single_sub)
    recall_matrix = np.zeros((n_story_segments, n_recall_segments), dtype=int)

    for idx_recall_segment, story_segment_indices in ratings_single_sub:
        for idx_story_segment in story_segment_indices:
            recall_matrix[idx_story_segment, idx_recall_segment] = 1
    return recall_matrix


def ratings_to_matrix(data: dict, sub_id: str) -> np.ndarray:
    num_story_segs = data["n_story_segments"]
    num_recall_segs = len(data["ratings"][sub_id])
    recall_matrix = np.zeros((num_story_segs, num_recall_segs), dtype=int)

    for recall_idx, story_indices in data["ratings"][sub_id]:
        for story_idx in story_indices:
            recall_matrix[story_idx, recall_idx] = 1

    return recall_matrix


def plot_recall_matrix(
    story_name: str,
    sub_id: str,
    recall_matrix: np.ndarray,
    title: str,
    output_dir: Path,
    measure_label: str = "binary",
):
    m, n = recall_matrix.shape
    fig, ax = plt.subplots(figsize=(int((n / m + 1) * 9), 9))

    im = ax.imshow(recall_matrix, cmap="Reds")
    ax.set_title(title)
    ax.set_xlabel("Recall segments")
    ax.set_ylabel("Story segments")
    ax.set_aspect(1)

    fig.colorbar(im, ax=ax, label=measure_label)
    fig.suptitle(f"{story_name} | {sub_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{story_name}_{sub_id}.png", bbox_inches="tight")
    plt.close(fig)


def human_csv_to_matrix(csv_path: Path, normalize: bool = True) -> np.ndarray:
    df = pd.read_csv(csv_path)

    # Ensure numeric
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0)

    # Preserve original ordering
    story_segments = df["story_segment"].unique()
    recall_segments = df["recall_segment"].unique()

    story_index = {s: i for i, s in enumerate(story_segments)}
    recall_index = {r: j for j, r in enumerate(recall_segments)}

    matrix = np.zeros((len(story_segments), len(recall_segments)))

    for _, row in df.iterrows():
        i = story_index[row["story_segment"]]
        j = recall_index[row["recall_segment"]]
        matrix[i, j] = row["quality_score"]

    if normalize:
        matrix = matrix / 6.0  # scale 0–6 → 0–1

    return matrix


def threshold_human_matrix(human_matrix: np.ndarray) -> np.ndarray:
    binary = (human_matrix > 0).astype(float)
    return binary


def inspect_recall_segments(
    recall_segments: list[str],
    story_segments: list[str],
    human_matrix: np.ndarray,
    llm_matrix: np.ndarray,
    recall_indices: list[int],
):
    for idx in recall_indices:
        print("\n" + "=" * 80)
        print(f"RECALL SEGMENT {idx}")
        print("-" * 80)
        print(recall_segments[idx].strip())

        print("\nModel matched story indices:")
        model_matches = np.where(llm_matrix[:, idx] > 0)[0]
        print(model_matches.tolist())

        print("\nHuman nonzero story indices (with scores):")
        human_matches = np.where(human_matrix[:, idx] > 0)[0]
        for i in human_matches:
            print(f"{i} (score={human_matrix[i, idx]:.2f})")

        print("\nStory text for model matches:")
        for i in model_matches:
            print(f"[{i}] {story_segments[i]}")

        print("\nStory text for human matches:")
        for i in human_matches:
            print(f"[{i}] {story_segments[i]}")
