from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import dotenv_values
from rich.console import Console
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

console = Console()
CONFIG = dotenv_values(".env")


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


def get_rm_filmfest_annotated(
    recalls_df: pd.DataFrame, transcript_df: pd.DataFrame
) -> tuple[list[str], list[np.ndarray]]:
    """Returns the recall matrices for each filmfest subject.

    Recall matrix: rows: story segments, columns: recall segments
    """

    M = transcript_df["seg_num"].max().item()
    # 1. convert multi-matched segments into multiple rows
    rms: list[np.ndarray] = list()
    sub_ids: list[str] = list()
    for sub_id, sub_recall_df in recalls_df.groupby("sub_id"):
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


def get_total_cross_entropy_x_given_y(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids_x: torch.Tensor,
    input_ids_y: torch.Tensor | None,
    is_rj: bool,
    verbose: bool = False,
) -> float:
    """Compute cross_entropy(LLM(X | Y), X)"""

    if input_ids_y is None:
        input_ids_yx = input_ids_x
        x_in_logits_start_idx = 0
        x_in_input_ids_start_idx = 1
    else:
        input_ids_yx = torch.cat([input_ids_y, input_ids_x], dim=1)
        x_in_logits_start_idx = input_ids_y.shape[1] - 1
        x_in_input_ids_start_idx = 0

    if verbose:
        if is_rj:
            console.print(
                "[italic red]H(E | R0 to Rj-1, Rj):[/italic red]"
                f" {tokenizer.decode(input_ids_yx[0])}"
            )
        else:
            console.print(
                "[italic red]H(E | R0 to Rj-1):[/italic red]"
                f" {tokenizer.decode(input_ids_yx[0])}"
            )

    with torch.no_grad():
        (logits_yx, _) = model(input_ids=input_ids_yx, return_dict=False)

    logits_yx = logits_yx.cpu()
    # choose the logits from x
    logits_x = (
        logits_yx[:, x_in_logits_start_idx:-1, :].cpu().view(-1, logits_yx.shape[-1])
    )
    labels_x = input_ids_x[:, x_in_input_ids_start_idx:].view(-1)
    cross_entropy = F.cross_entropy(logits_x, labels_x, reduction="none")
    return cross_entropy.numpy().sum()


def mutual_information_normalized(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    ids_story_segments: list[torch.Tensor],
    r0_to_r_j_minus_1: torch.Tensor,
    rj: torch.Tensor,
    ids_story_segments_with_bos: list[torch.Tensor] | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the normalized mutual information"""

    # TODO: use batch processing
    # TODO: do some serious caching here
    mi_for_j = np.zeros(len(ids_story_segments))
    for idx_story_segment, ids_story_segment in enumerate(ids_story_segments):
        if len(r0_to_r_j_minus_1) == 0:
            assert ids_story_segments_with_bos is not None
            H_e_given_r0_to_r_j_minus_1 = get_total_cross_entropy_x_given_y(
                model,
                tokenizer,
                ids_story_segments_with_bos[idx_story_segment],
                input_ids_y=None,
                is_rj=False,
                verbose=verbose,
            )
        else:
            H_e_given_r0_to_r_j_minus_1 = get_total_cross_entropy_x_given_y(
                model,
                tokenizer,
                ids_story_segment,
                input_ids_y=r0_to_r_j_minus_1,
                is_rj=False,
                verbose=verbose,
            )

        H_e_given_r0_to_r_j = get_total_cross_entropy_x_given_y(
            model,
            tokenizer,
            ids_story_segment,
            input_ids_y=torch.cat([r0_to_r_j_minus_1, rj], dim=1),
            is_rj=True,
            verbose=verbose,
        )

        mi_for_j[idx_story_segment] = max(
            0, 1 - (H_e_given_r0_to_r_j / H_e_given_r0_to_r_j_minus_1)
        )
        if verbose:
            console.print(
                f"> MUTUAL INFORMATION: {mi_for_j[idx_story_segment]}",
                style="bold yellow",
            )

    return mi_for_j


def compute_rm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    story_segments: list[str],
    recall_segments: list[str],
    verbose: bool = False,
) -> np.ndarray:
    """Compute the recall matrix"""

    # encode all story segments
    ids_story_segments: list[torch.Tensor] = [  # type: ignore
        tokenizer.encode(story_segment, return_tensors="pt", add_special_tokens=False)
        for story_segment in story_segments
    ]
    # for r0, need to add bos token
    ids_story_segments_with_bos: list[torch.Tensor] = [  # type: ignore
        tokenizer.encode(story_segment, return_tensors="pt", add_special_tokens=True)
        for story_segment in story_segments
    ]

    r0_to_r_j_minus_1_list: list[torch.Tensor] = list()
    M = len(ids_story_segments)
    N = len(recall_segments)
    recall_matrix = np.empty((M, N))
    for idx_recall_segment, recall_segment in enumerate(recall_segments):
        if idx_recall_segment == 0:
            # r0 will be used as the first segment in one of the computations
            # -> start with bos token
            rj: torch.Tensor = tokenizer.encode(recall_segment, return_tensors="pt")  # type: ignore

            recall_matrix[:, idx_recall_segment] = mutual_information_normalized(
                model=model,
                tokenizer=tokenizer,
                ids_story_segments=ids_story_segments,
                r0_to_r_j_minus_1=torch.empty(0, dtype=ids_story_segments[0].dtype),
                rj=rj,
                ids_story_segments_with_bos=ids_story_segments_with_bos,
                verbose=verbose,
            )

        else:
            # rj is consecutive
            # -> do not add bos token
            rj: torch.Tensor = tokenizer.encode(  # type: ignore
                recall_segment, return_tensors="pt", add_special_tokens=False
            )

            recall_matrix[:, idx_recall_segment] = mutual_information_normalized(
                model=model,
                tokenizer=tokenizer,
                ids_story_segments=ids_story_segments,
                r0_to_r_j_minus_1=torch.cat(r0_to_r_j_minus_1_list, dim=1),
                rj=rj,
                verbose=verbose,
            )

        r0_to_r_j_minus_1_list.append(rj)
    return recall_matrix


def get_filmfest_rm_mi(
    model_name: str,
    recalls_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    verbose: bool = False,
) -> tuple[list[str], list[str], list[np.ndarray]]:
    model = AutoModelForCausalLM.from_pretrained(model_name, token=CONFIG["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=CONFIG["HF_TOKEN"])

    sub_ids: list[str] = list()
    movie_names: list[str] = list()
    recall_matrices_mi: list[np.ndarray] = list()

    for sub_id, sub_df in recalls_df.groupby("sub_id"):
        for story_id, story_df in transcript_df.groupby("movie_num"):
            story_segments = (
                story_df.groupby("seg_num")["text"].apply(" ".join).tolist()
            )
            recall_segments = sub_df.loc[
                sub_df["movie_num"] == story_id, "text"
            ].tolist()
            mi_recall_matrix = compute_rm(
                model=model,
                tokenizer=tokenizer,
                story_segments=story_segments,
                recall_segments=recall_segments,
                verbose=verbose,
            )
            sub_ids.append(sub_id)  # type: ignore
            movie_names.append(story_id)  # type: ignore
            recall_matrices_mi.append(mi_recall_matrix)
    return sub_ids, movie_names, recall_matrices_mi


if __name__ == "__main__":
    recalls_df = load_recalls(dataset="filmfest", n_subjects=3)
    transcript_df = load_transcripts(dataset="filmfest")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    sub_ids, movie_names, recall_matrices_mi = get_filmfest_rm_mi(
        model_name=model_name,
        recalls_df=recalls_df,
        transcript_df=transcript_df,
        verbose=True,
    )

    story_indices = transcript_df.groupby("seg_num")["movie_num"].first().to_numpy()
    sub_ids, rms = get_rm_filmfest_annotated(recalls_df, transcript_df)
    plot_rms(story_indices=story_indices, sub_titles=sub_ids, rms=rms)
