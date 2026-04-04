# ruff: noqa: E501
import re
from typing import Callable

from rmatch import get_logger

log = get_logger(__name__)


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


def prompt_default(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
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

    # 1. format story segments
    story_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(recall_segments, target_idx, window_size)

    # 3. build parser
    def parse_output(raw: str) -> set[int] | None:
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
        f"""You are an expert annotator for episodic memory research. Your task is to match which numbered story elements a participant's recall segment refers to.

Matching guidelines:
- Match based on semantic content, not surface wording. Recall is often paraphrased.
- A single recall segment may reference multiple story elements. List all that apply.
- Do NOT match inferences or distortions unless a specific element is clearly referenced.
- Match only what is explicitly or clearly implicitly referenced. Do not match based on vague thematic similarity.
- The surrounding recall context aids comprehension only. Match based solely on the <target> segment.

Return matching story element numbers between <> tags, or <NONE> if none match.

<EXAMPLE>
<story_elements>
1. Sarah walked into the coffee shop.
2. She ordered a latte.
3. The barista spilled the drink.
4. Sarah laughed and said it was fine.
5. Sarah sat down at the empty table across the room.
</story_elements>

<recall_context>
Sarah went to a order a coffee.
<target>She laughed when the barista spilled the drink, because she is a kind person</target>
</recall_context>
Story segments that match the <target> segment:
<3,4>
</EXAMPLE>

Now, given the actual story and context, return the matching story elements:
<story_elements>
{story_segments_str}
</story_elements>

<recall_context>
{window_str}
</recall_context>
Story segments that match the <target> segment:
""",
        parse_output,
    )


def build_prompt_alternative(
    story_segments: list[str],
    recall_segments: list[str],
    target_idx: int,
    window_size: int,
    verbose_errors: bool = False,
) -> tuple[str, Callable[[str], set[int] | None]]:
    """Return prompt and parser."""

    # 1. full story
    story_str = " ".join(story_segments)

    # 1. format story segments
    stor_segments_str = format_story_segments(story_segments)

    # 2. build recall window
    window_str = build_recall_window(
        recall_segments, target_idx, window_size, start_marker=">>> ", end_marker=" <<<"
    )

    # 3. build parser
    def parse_output(raw: str) -> set[int] | None:
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
{story_str}

It can be broken down into the following independent pieces of information:
{stor_segments_str}

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
<3, 7, 12>
Story segments that match the <target> segment:
""",
        parse_output,
    )
