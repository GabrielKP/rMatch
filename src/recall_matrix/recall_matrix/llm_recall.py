import re
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from matplotlib import pyplot as plt
from openai import OpenAI
from tqdm import tqdm

from recall_matrix.load import load_story_recall_segments

# TODO: look @ rate_binary for output format
# 5 nano, 4o mini, 4.1 nano (cheap ones)
STORY = "pieman"
SUBJECT = "sub-001"


def build_message(recall_segment: str, story: str, story_segments: str) -> str:
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
    return message


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


def compute_matrix() -> np.ndarray:
    api = dotenv_values(".env")["OPENAI_API_KEY"]
    client = OpenAI(api_key=api)

    data = load_story_recall_segments(
        story_name=STORY,
        story_segment_method="sentences_corrected",
        recall_segment_method="sentences",
        sub_ids=[SUBJECT],
    )[0]

    sub_id, story_segments, recall_segments = data[0]

    story = format_story(story_segments)
    story_segments_formatted = format_story_segments(story_segments)

    num_story = len(story_segments)
    num_recall = len(recall_segments)

    recall_matrix = np.zeros((num_story, num_recall), dtype=int)

    for col, recall in enumerate(
        tqdm(recall_segments, desc="Rating recalls", unit="clause")
    ):
        query = build_message(recall, story, story_segments_formatted)
        response = client.responses.create(model="gpt-5-nano", input=query)
        # tqdm.write(str(response.output_text))
        parsed_response = parse(response.output_text)

        for row in parsed_response:
            recall_matrix[row - 1, col] = 1

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


if __name__ == "__main__":
    matrix = compute_matrix()
    plot_recall_matrix(
        STORY, SUBJECT, matrix, "LLM Recall Proto", Path("outputs/tests/llm")
    )
