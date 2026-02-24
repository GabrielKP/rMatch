import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from matplotlib import pyplot as plt
from openai import OpenAI
from tqdm import tqdm

from recall_matrix.load import load_story_recall_segments

# TODO: test diff models 5 nano, 4o mini, 4.1 nano (cheap ones)
# TODO: give recall context in prompt
# TODO: try prompt w/ recall context
STORY = "pieman"
SUBJECT = "sub-003"


def build_message(
    recall_segment: str, story: str, story_segments: str, window: str | None = None
) -> str:
    if window is None:
        message = f"""This is the original text:
    {story}

    It can be broken down into the following independent pieces of information:
    {story_segments}

    Here is a single clause from a participant's recall of the story:
    "{recall_segment}"

    Which of the numbered information pieces above are expressed by this recall clause?
    List all applicable numbers. If none apply, return the string "<NONE>".

    Return ONLY a set of numbers in <>, for example:
    <3, 7, 12>
    """
    else:
        message = f"""This is the original story:
    {story}

    It can be broken down into the following independent pieces of information:
    {story_segments}

    Below is a window of consecutive clauses from a participant's recall.
    The TARGET clause is marked with >>> <<<.
    The other clauses are provided _only as context_.

    Recall window:
    {window}

    Which of the numbered information pieces are expressed by the >>> TARGET <<< clause?

    Use the surrounding clauses only to resolve references (e.g., pronouns),
    but DO NOT attribute information expressed only in neighboring clauses.
    If a numbered information piece is not explicitly expressed in the TARGET
    clause itself, do NOT include it.

    If none apply, return "<NONE>".

    Return ONLY a set of numbers in angle brackets, for example:
    <3, 7, 12>"""

    return message


def build_recall_window(
    recall_segments: list[str], idx: int, window_size: int = 5
) -> str:
    start = max(0, idx - window_size)
    end = min(len(recall_segments), idx + window_size + 1)

    lines = []
    for i in range(start, end):
        clause = recall_segments[i].strip()
        if i == idx:
            lines.append(f">>> {clause} <<<")
        else:
            lines.append(f"{clause}")

    return "\n".join(lines)


def format_story_segments(story_segments: list[str]) -> str:
    return "\n".join(f"{i+1}. {seg.strip()}" for i, seg in enumerate(story_segments))


def format_story(story_segments: list[str]) -> str:
    return " ".join(seg.strip() for seg in story_segments)


def parse(raw: str) -> set[int]:
    raw = raw.strip()

    if "<NONE>" in raw:
        return set()

    match = re.search(r"<([0-9,\s]+)>", raw)
    if not match:
        return set()  # silent fail?

    parsed_set = {
        int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
    }

    return parsed_set


def rate_llm_binary(story_name: str, model_name: str = "gpt-5-nano") -> dict:
    api = dotenv_values(".env")["OPENAI_API_KEY"]
    client = OpenAI(api_key=api)

    data, story_segment_method, recall_segment_method = load_story_recall_segments(
        story_name=story_name,
        story_segmentation_method="sentences_corrected",
        recall_segmentation_method="sentences",
        sub_ids=[SUBJECT],
    )

    output_dict = {
        "param_str": f"llm_binary_{model_name}",
        "story_name": story_name,
        "story_segment_method": story_segment_method,
        "recall_segment_method": recall_segment_method,
        "model_name": model_name,
        "output_scores": False,
    }

    first_sub_id, segs, _ = data[0]
    output_dict["num_story_segments"] = len(segs)

    ratings: dict[str, list[tuple[int, list[int]]]] = {}

    for sub_id, story_segments, recall_segments in data:
        story = format_story(story_segments)
        story_segments_formatted = format_story_segments(story_segments)

        sub_ratings: list[tuple[int, list[int]]] = []

        for recall_idx, recall in enumerate(
            tqdm(recall_segments, desc="Rating recalls", unit="clause")
        ):
            recall_context = build_recall_window(recall_segments, recall_idx)
            query = build_message(
                recall, story, story_segments_formatted, recall_context
            )
            response = client.responses.create(model=model_name, input=query)
            # tqdm.write(str(response.output_text))
            parsed_response = parse(response.output_text)

            story_indices = sorted(idx - 1 for idx in parsed_response)
            sub_ratings.append((recall_idx, story_indices))

        ratings[sub_id] = sub_ratings

    output_dict["ratings"] = ratings

    output_path = (
        Path("data")
        / "stories-and-recalls"
        / story_name
        / "ratings"
        / f"{output_dict['param_str']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_dict, f)
        f.write("\n")

    return output_dict


def ratings_to_matrix(data: dict, sub_id: str) -> np.ndarray:
    num_story_segs = data["num_story_segments"]
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


# TODO: smth smth smth argparse, yoink from rate_binary (main)
if __name__ == "__main__":
    model = "gpt-5.2"
    ratings = rate_llm_binary(STORY, model_name=model)
    llm_matrix = ratings_to_matrix(ratings, SUBJECT)
    human_csv_path = Path(
        f"data/stories-and-recalls/{STORY}/recall_quality/human/"
        f"{SUBJECT}-DA-ssm_sentences_corrected-rsm_sentences.csv"
    )

    human_matrix_norm = human_csv_to_matrix(human_csv_path, normalize=True)
    human_matrix_bin = threshold_human_matrix(human_matrix_norm)

    llm_matrix = llm_matrix.astype(float)

    diff = np.abs(human_matrix_bin - llm_matrix)

    disagreement_mask = diff.sum(axis=0) > 0
    disagreed_indices = np.where(disagreement_mask)[0].tolist()

    data, _, _ = load_story_recall_segments(
        story_name=STORY,
        story_segmentation_method="sentences_corrected",
        recall_segmentation_method="sentences",
        sub_ids=[SUBJECT],
    )

    _, story_segments, recall_segments = data[0]

    inspect_recall_segments(
        recall_segments=recall_segments,
        story_segments=story_segments,
        human_matrix=human_matrix_norm,
        llm_matrix=llm_matrix,
        recall_indices=disagreed_indices,
    )

    plot_recall_matrix(
        "pieman",
        f"{SUBJECT}-diff-DA",
        diff,
        "Absolute Difference (Human vs Model)",
        Path("outputs/tests/comparison"),
        measure_label="|human - model|",
    )

    plot_recall_matrix(
        story_name="pieman",
        sub_id=f"{SUBJECT}-human",
        recall_matrix=human_matrix_norm,
        title="Human Ratings (Normalized)",
        output_dir=Path("outputs/tests/human"),
        measure_label="normalized score (0-1)",
    )
    plot_recall_matrix(
        STORY,
        SUBJECT,
        llm_matrix,
        f"{model} Recall Proto (with context)",
        Path("outputs/tests/llm"),
    )
