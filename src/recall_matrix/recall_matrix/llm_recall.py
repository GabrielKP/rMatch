import re

import numpy as np
from openai import OpenAI

# TODO: LINK TO GPT
from recall_matrix.load import (
    load_cyoa_recall_matrix_human_binary,
    load_cyoa_story_recall_segments,
)

# TODO: figure data out lol
STORY = "story name smth smth smth"
SUBJECT = "sample subject name"


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
    data = load_cyoa_story_recall_segments([STORY])

    for story_name, subj, story_segments, recall_segments in data:
        if story_name == STORY and subj == SUBJECT:
            break

    story = format_story(story_segments)
    story_segments_formatted = format_story_segments(story_segments)

    num_story = len(story_segments)
    num_recall = len(recall_segments)

    recall_matrix = np.zeros((num_story, num_recall), dtype=int)

    for col, recall in enumerate(recall_segments):
        query = build_message(recall, story, story_segments_formatted)
        # TODO: contact llm here
        response = model.message(query)
        parsed_response = parse(response)

        for row in parsed_response:
            recall_matrix[row - 1, col] = 1

    return recall_matrix
