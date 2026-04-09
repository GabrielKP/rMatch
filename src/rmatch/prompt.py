"""Prompts:

primary:
- includes full story, segmented story, chain of thought prompt, recall window

primary_no_story:
- segmented story, chain of thought, recall window (useful for long stories, prompt may get too long)

primary_no_cot:
- includes full story, segmented story, recall window but no chain of thought (ablation test)

primary_no_story_no_cot:
- only segmented story, recall window (ablation test)
"""

# ruff: noqa: E501
import re
from typing import Callable

from rmatch import get_logger

log = get_logger(__name__)


def get_prompt_and_parser(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
    prompt: str | None = None,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """Return prompt and parser.

    Each prompt returns its own parser, because different prompts may
    require different parsing logic. Input to the parser is the prompt
    plus the text output from the model.

    Returns
    -------
    prompt: str
        prompt to be sent to the model
    parser: Callable[[str], set[int] | None]
        parser to parse the output from the model
    """
    if prompt is None:
        prompt = "primary"

    if prompt == "primary":
        return prompt_primary(
            story_segments, recall_segments, target_idx, window_size, verbose_errors
        )
    elif prompt == "primary_no_story":
        return prompt_primary_no_story(
            story_segments, recall_segments, target_idx, window_size, verbose_errors
        )
    elif prompt == "primary_no_cot":
        return prompt_primary_no_cot(
            story_segments, recall_segments, target_idx, window_size, verbose_errors
        )
    elif prompt == "primary_no_story_no_cot":
        return prompt_primary_no_story_no_cot(
            story_segments, recall_segments, target_idx, window_size, verbose_errors
        )
    elif prompt == "secondary":
        return prompt_secondary(
            story_segments, recall_segments, target_idx, window_size, verbose_errors
        )
    else:
        raise ValueError(f"Invalid prompt type: {prompt}")


### Helper functions ###


def build_recall_window(
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    start_marker: str = "<target>",
    end_marker: str = "</target>",
) -> str:
    if window_size == 0:
        return f"{start_marker}{recall_segments[target_idx]}{end_marker}"

    start = max(0, target_idx - window_size)
    end = min(len(recall_segments), target_idx + window_size + 1)

    lines = []
    for i in range(start, end):
        clause = recall_segments[i].strip()
        if i == target_idx:
            lines.append(f"{start_marker}{clause}{end_marker}")
        else:
            lines.append(f"{clause}")

    return "\n".join(lines)


def format_story_segments(story_segments: list[str]) -> str:
    return "\n".join(f"{i + 1}. {seg.strip()}" for i, seg in enumerate(story_segments))


### Primary prompts ###


def prompt_primary(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """Primary prompt: includes full story, segmented story, and chain of thought."""

    # 1. format story segments
    story_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(
        recall_segments, target_idx, window_size, start_marker=">>> ", end_marker=" <<<"
    )

    # 3. full story
    story_sr = " ".join(seg.strip() for seg in story_segments)

    def _parse_output(raw: str) -> set[int] | None:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            if verbose_errors:
                log.warning(f"failed to parse model output:\n{raw}")
            return None

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    return (
        f"""This is the original story:
{story_sr}

It can be broken down into the following independent pieces of information:
{story_segments_str}

Below is a window of consecutive clauses from a participant's recall.
The TARGET clause is marked with >>> <<<.
The other clauses are provided _only as context_.

Recall window:
{window_str}

Which of the numbered information pieces are
expressed by the >>> TARGET <<< clause?

Use the surrounding clauses only to resolve references (e.g., pronouns),
but DO NOT attribute information expressed only in neighboring clauses.
If a numbered information piece is not explicitly expressed in the TARGET
clause itself, do NOT include it.

If none apply, return "<NONE>".

First reason about the TARGET clause and which story elements it maps to.

Then return a set of numbers in angle brackets, for example:
<3, 7, 12>""",
        _parse_output,
    )


def prompt_primary_no_story(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """No full story: segmented story, chain of thought, recall window."""

    # 1. format story segments
    story_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(
        recall_segments, target_idx, window_size, start_marker=">>> ", end_marker=" <<<"
    )

    def _parse_output(raw: str) -> set[int] | None:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            if verbose_errors:
                log.warning(f"failed to parse model output:\n{raw}")
            return None

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    return (
        f"""This is the original story broken down into the following independent pieces of information:
{story_segments_str}

Below is a window of consecutive clauses from a participant's recall.
The TARGET clause is marked with >>> <<<.
The other clauses are provided _only as context_.

Recall window:
{window_str}

Which of the numbered information pieces are
expressed by the >>> TARGET <<< clause?

Use the surrounding clauses only to resolve references (e.g., pronouns),
but DO NOT attribute information expressed only in neighboring clauses.
If a numbered information piece is not explicitly expressed in the TARGET
clause itself, do NOT include it.

If none apply, return "<NONE>".

First reason about the TARGET clause and which story elements it maps to.

Then return a set of numbers in angle brackets, for example:
<3, 7, 12>""",
        _parse_output,
    )


def prompt_primary_no_cot(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """No chain of thought: full story, segmented story, recall window."""

    # 1. format story segments
    story_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(
        recall_segments, target_idx, window_size, start_marker=">>> ", end_marker=" <<<"
    )

    # 3. full story
    story_sr = " ".join(seg.strip() for seg in story_segments)

    def _parse_output(raw: str) -> set[int] | None:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            if verbose_errors:
                log.warning(f"failed to parse model output:\n{raw}")
            return None

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    return (
        f"""This is the original story:
{story_sr}

It can be broken down into the following independent pieces of information:
{story_segments_str}

Below is a window of consecutive clauses from a participant's recall.
The TARGET clause is marked with >>> <<<.
The other clauses are provided _only as context_.

Recall window:
{window_str}

Which of the numbered information pieces are
expressed by the >>> TARGET <<< clause?

Use the surrounding clauses only to resolve references (e.g., pronouns),
but DO NOT attribute information expressed only in neighboring clauses.
If a numbered information piece is not explicitly expressed in the TARGET
clause itself, do NOT include it.

If none apply, return "<NONE>".

Return ONLY a set of numbers in angle brackets, for example:
<3, 7, 12>""",
        _parse_output,
    )


def prompt_primary_no_story_no_cot(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """No story, no chain of thought: Just segmented story + recall window."""

    # 1. format story segments
    story_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(
        recall_segments, target_idx, window_size, start_marker=">>> ", end_marker=" <<<"
    )

    def _parse_output(raw: str) -> set[int] | None:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            if verbose_errors:
                log.warning(f"failed to parse model output:\n{raw}")
            return None

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    return (
        f"""This is the original story broken down into the following independent pieces of information:
{story_segments_str}

Below is a window of consecutive clauses from a participant's recall.
The TARGET clause is marked with >>> <<<.
The other clauses are provided _only as context_.

Recall window:
{window_str}

Which of the numbered information pieces are
expressed by the >>> TARGET <<< clause?

Use the surrounding clauses only to resolve references (e.g., pronouns),
but DO NOT attribute information expressed only in neighboring clauses.
If a numbered information piece is not explicitly expressed in the TARGET
clause itself, do NOT include it.

If none apply, return "<NONE>".

Return ONLY a set of numbers in angle brackets, for example:
<3, 7, 12>""",
        _parse_output,
    )


def prompt_secondary(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """Secondary version of the prompt."""

    # 1. format story segments
    story_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(recall_segments, target_idx, window_size)

    # 3. build parser
    def _parse_output(raw: str) -> set[int] | None:
        raw = raw.strip()

        if "<NONE>" in raw:
            return set()

        match = re.search(r"<([0-9,\s]+)>", raw)
        if not match:
            if verbose_errors:
                log.warning(f"failed to parse model output:\n{raw}")
            return None

        parsed_set = {
            int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
        }

        return parsed_set

    # format story
    story_str = " ".join(seg.strip() for seg in story_segments)

    return (
        f"""You are an expert annotator for episodic memory research. Your task is to match which numbered story elements a participant's recall segment refers to.

Matching guidelines:
- Match based on semantic content, not surface wording. Recall is often paraphrased.
- A single recall segment may reference multiple story elements. List all that apply.
- Do NOT match inferences or distortions unless a specific element is clearly referenced.
- Match only what is explicitly or clearly implicitly referenced. Do not match based on vague thematic similarity.
- The surrounding recall context aids comprehension only. Match based solely on the <target> segment.


First, in <reasoning> tags, briefly state what the target segment conveys and which story elements it maps to. Then list the matched element numbers.

Return matching story element numbers between <> tags, or <NONE> if none match.


Here is the full story:
<story>
{story_str}
</story>

It can be broken down into the following independent pieces of information:
<story_elements>
{story_segments_str}
</story_elements>

<recall_context>
{window_str}
</recall_context>
Thoughts and story segments that match the <target> segment:
""",
        _parse_output,
    )
