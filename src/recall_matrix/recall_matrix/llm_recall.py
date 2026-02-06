import json
import re
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from matplotlib import pyplot as plt
from openai import OpenAI
from tqdm import tqdm

from recall_matrix.load import load_story_recall_segments

# TODO: test diff models 5 nano, 4o mini, 4.1 nano (cheap ones)
# TODO: give recall context in prompt
# TODO: try prompt w/ recall context
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


def rate_llm_binary(story_name: str, model_name: str = "gpt-5-nano") -> dict:
    api = dotenv_values(".env")["OPENAI_API_KEY"]
    client = OpenAI(api_key=api)

    data, story_segment_method, recall_segment_method = load_story_recall_segments(
        story_name=story_name,
        story_segment_method="sentences_corrected",
        recall_segment_method="sentences",
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
            query = build_message(recall, story, story_segments_formatted)
            response = client.responses.create(model=model_name, input=query)
            # tqdm.write(str(response.output_text))
            parsed_response = parse(response.output_text)

            story_indicies = sorted(idx - 1 for idx in parsed_response)
            sub_ratings.append((recall_idx, story_indicies))

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

    for recall_idx, story_indicies in data["ratings"][sub_id]:
        for story_idx in story_indicies:
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


# TODO: smth smth smth argparse, yoink from rate_binary (main)
if __name__ == "__main__":
    ratings = rate_llm_binary(STORY, model_name="gpt-4.1")
    matrix = ratings_to_matrix(ratings, SUBJECT)
    plot_recall_matrix(
        STORY, SUBJECT, matrix, "LLM Recall Proto", Path("outputs/tests/llm")
    )
