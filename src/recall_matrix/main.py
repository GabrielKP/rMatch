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


def get_total_cross_entropy_x_given_y(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_str_x: str,
    input_str_y: str,
    is_rj: bool,
    verbose: bool = False,
) -> float:
    """Compute cross_entropy(LLM(X | Y), X)"""

    # tokenize
    if len(input_str_y) > 0:
        input_str = f"{input_str_y} {input_str_x}"
        idx_char_x = len(input_str_y) + 1
    else:
        input_str = input_str_x
        idx_char_x = 0

    batch_encoding = tokenizer(
        input_str, return_tensors="pt", return_offsets_mapping=True
    )
    input_ids: torch.Tensor = batch_encoding.input_ids
    offset_mapping: torch.Tensor = batch_encoding.offset_mapping

    idx_id_x = min(np.where(offset_mapping[:, :, -1] > idx_char_x)[1])

    if verbose:
        prefix = (
            "[italic red]H(E | R0 to Rj-1, Rj):[/italic red]"
            if is_rj
            else "[italic red]H(E | R0 to Rj-1):[/italic red]"
        )
        decode_str_y = tokenizer.decode(input_ids[0, :idx_id_x])
        decode_str_x = tokenizer.decode(input_ids[0, idx_id_x:])

        console.print(
            f"{prefix} [white]{decode_str_y}[/white][green]{decode_str_x}[/green]"
        )

    with torch.no_grad():
        (logits, _) = model(input_ids=input_ids, return_dict=False)

    logits = logits.cpu()
    # choose only the logits from x

    # input_ids: [... ,      y_j,      x_0,      x_1, ...,    x_n-1,      x_n]
    # labels_x :                        ^                                   ^
    labels_x = input_ids[:, idx_id_x:]
    # logits   : [... , pred_x_0, pred_x_1, pred_x_2, ..., pred_x_n, pred_x_n+1]
    # logits_x :             ^                                    ^
    logits_x = logits[:, idx_id_x - 1 : -1, :].cpu()

    logits_x_flat = logits_x.view(-1, logits_x.shape[-1])
    labels_x_flat = labels_x.view(-1)

    cross_entropy = F.cross_entropy(logits_x_flat, labels_x_flat, reduction="none")
    return cross_entropy.numpy().sum()


def mutual_information_with_history(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    normalize: bool,
    story_segments: list[str],
    r0_to_r_j_minus_1: list[str],
    rj: str,
    rj_idx: int,  # just for debugging
    verbose: bool = False,
) -> np.ndarray:
    """Compute the normalized mutual information"""

    # TODO: Can do some serious optimization: batching, caching, ..
    mi_for_j = np.zeros(len(story_segments))
    for idx_story_segment, story_segment in enumerate(story_segments):
        H_e_given_r0_to_r_j_minus_1 = get_total_cross_entropy_x_given_y(
            model=model,
            tokenizer=tokenizer,
            input_str_x=story_segment,
            input_str_y=" ".join(r0_to_r_j_minus_1),
            is_rj=False,
            verbose=verbose,
        )

        H_e_given_r0_to_r_j = get_total_cross_entropy_x_given_y(
            model=model,
            tokenizer=tokenizer,
            input_str_x=story_segment,
            input_str_y=" ".join([*r0_to_r_j_minus_1, rj]),
            is_rj=True,
            verbose=verbose,
        )

        if normalize:
            mi_for_j[idx_story_segment] = max(
                0, 1 - (H_e_given_r0_to_r_j / H_e_given_r0_to_r_j_minus_1)
            )
        else:
            mi_for_j[idx_story_segment] = max(
                0, (H_e_given_r0_to_r_j_minus_1 - H_e_given_r0_to_r_j)
            )
        if verbose:
            console.print(
                (
                    f"> MI(E_{idx_story_segment}, R_{rj_idx})"
                    f" = {mi_for_j[idx_story_segment]}"
                ),
                style="bold yellow",
            )

    return mi_for_j


def mutual_information(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    mutual_information_method: str,
    normalize: bool,
    story_segments: list[str],
    r0_to_r_j_minus_1: list[str],
    rj: str,
    rj_idx: int,
    verbose: bool = False,
) -> np.ndarray:
    """Compute the mutual information"""

    if mutual_information_method == "with_history":
        return mutual_information_with_history(
            model=model,
            tokenizer=tokenizer,
            normalize=normalize,
            story_segments=story_segments,
            r0_to_r_j_minus_1=r0_to_r_j_minus_1,
            rj=rj,
            rj_idx=rj_idx,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Mutual information method {mutual_information_method} not found"
        )


def compute_rm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    mutual_information_method: str,
    normalize: bool,
    story_segments: list[str],
    recall_segments: list[str],
    verbose: bool = False,
) -> np.ndarray:
    """Compute the recall matrix"""

    M = len(story_segments)
    N = len(recall_segments)
    recall_matrix = np.empty((M, N))
    for idx_recall_segment, recall_segment in enumerate(recall_segments):
        recall_matrix[:, idx_recall_segment] = mutual_information(
            model=model,
            tokenizer=tokenizer,
            mutual_information_method=mutual_information_method,
            normalize=normalize,
            story_segments=story_segments,
            r0_to_r_j_minus_1=recall_segments[:idx_recall_segment],
            rj=recall_segment,
            rj_idx=idx_recall_segment,
            verbose=verbose,
        )

    return recall_matrix


def get_filmfest_rm_mi(
    model_name: str,
    recalls_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    mutual_information_method: str,
    mutual_information_normalize: bool,
    verbose: bool = False,
) -> tuple[list[str], list[int], list[np.ndarray]]:
    model = AutoModelForCausalLM.from_pretrained(model_name, token=CONFIG["HF_TOKEN"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=CONFIG["HF_TOKEN"])

    sub_ids: list[str] = list()
    movie_nums: list[int] = list()
    recall_matrices_mi: list[np.ndarray] = list()

    for sub_id, sub_df in recalls_df.groupby("sub_id"):
        for movie_num, recall_df in sub_df.groupby("movie_num"):
            story_df = transcript_df[transcript_df["movie_num"] == movie_num]
            story_segments = (
                story_df.groupby("seg_num")["text"].apply(" ".join).tolist()
            )
            recall_segments = sub_df.loc[
                sub_df["movie_num"] == movie_num, "text"
            ].tolist()
            mi_recall_matrix = compute_rm(
                model=model,
                tokenizer=tokenizer,
                mutual_information_method=mutual_information_method,
                normalize=mutual_information_normalize,
                story_segments=story_segments,
                recall_segments=recall_segments,
                verbose=verbose,
            )
            sub_ids.append(sub_id)  # type: ignore
            movie_nums.append(movie_num)  # type: ignore
            recall_matrices_mi.append(mi_recall_matrix)
    return sub_ids, movie_nums, recall_matrices_mi


def plot_rms_comparison(
    conditions: list[str],
    sub_ids_conditions: list[list[str]],
    movie_nums_conditions: list[list[int]],
    rms_conditions: list[list[np.ndarray]],
):
    """Plot the recall matrices comparison"""

    comparison_name = "-".join(conditions)
    output_dir = Path("outputs") / "plots" / "rms_comparison" / comparison_name
    print(f"Saving plots to {str(output_dir)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # need to rearrange the rms such that condition is the inner loop
    paired_sub_ids = list(zip(*sub_ids_conditions))
    paired_movie_nums = list(zip(*movie_nums_conditions))
    paired_rms = list(zip(*rms_conditions))

    for paired_sub_id, paired_movie_num, paired_rm in zip(
        paired_sub_ids, paired_movie_nums, paired_rms
    ):
        fig, axes = plt.subplots(
            1,
            len(conditions),
            figsize=(7 * len(conditions), 10),
        )
        prev_sub_id = "-1"
        prev_movie_num = "-1"
        for idx, (sub_id, movie_num, rm, condition) in enumerate(
            zip(paired_sub_id, paired_movie_num, paired_rm, conditions)
        ):
            if idx == 0:
                prev_sub_id = sub_id
            else:
                assert prev_sub_id == sub_id
            if idx == 0:
                prev_movie_num = movie_num
            else:
                assert prev_movie_num == movie_num

            ax = axes[idx]
            # color_matrix = np.ones_like(rm)
            # ax.imshow(color_matrix, alpha=rm, cmap="Reds")
            ax.imshow(rm, cmap="Reds")
            ax.set_facecolor("black")
            ax.set_title(condition)
            ax.set_xlabel("Recall segments")
            ax.set_ylabel("Story segments")
            ax.set_aspect(1)

        fig.tight_layout()
        output_path = output_dir / str(prev_movie_num) / f"{prev_sub_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)


if __name__ == "__main__":
    sub_ids = ["sub-01"]
    movie_nums = [1, 2]
    recalls_df = load_recalls(
        dataset="filmfest", sub_ids=sub_ids, movie_nums=movie_nums
    )
    transcript_df = load_transcripts(dataset="filmfest")

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    sub_ids_mi, movie_nums_mi, recall_matrices_mi = get_filmfest_rm_mi(
        model_name=model_name,
        recalls_df=recalls_df,
        transcript_df=transcript_df,
        mutual_information_method="with_history",
        mutual_information_normalize=False,
        verbose=True,
    )

    sub_ids_binary, movie_nums_binary, rms_binary = get_filmfest_rm_binary(
        recalls_df, transcript_df
    )

    plot_rms_comparison(
        conditions=["binary_human", "mi_llama_1b_instruct"],
        sub_ids_conditions=[sub_ids_binary, sub_ids_mi],
        movie_nums_conditions=[movie_nums_binary, movie_nums_mi],
        rms_conditions=[rms_binary, recall_matrices_mi],
    )
