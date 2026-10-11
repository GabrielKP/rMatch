import re
from collections import defaultdict

from rmatch import get_logger, matchlist_type, parser_type
from rmatch.utils import build_recall_window, format_story_segments

log = get_logger(__name__)


DEFAULT_PROMPT = """This is the original story:
{story_full}

It can be broken down into the following independent pieces of information:
{story_segments}

Below is a window of consecutive clauses from a participant's recall.
The TARGET clause is marked with >>> <<<.
The other clauses are provided _only as context_.

Recall window:
{recall_window}

Which of the numbered information pieces are
expressed by the >>> TARGET <<< clause?

Use the surrounding clauses only to resolve references (e.g., pronouns),
but DO NOT attribute information expressed only in neighboring clauses.
If a numbered information piece is not explicitly expressed in the TARGET
clause itself, do NOT include it.

If none apply, return "<NONE>".

First reason about the TARGET clause and which story elements it maps to.

Then return a set of numbers in angle brackets, for example:
<3, 7, 12>"""


def _parse_output(raw: str) -> set[int] | None:
    raw = raw.strip()

    if "<NONE>" in raw:
        return set()

    match = re.search(r"<([0-9,\s]+)>", raw)
    if not match:
        log.debug(f"failed to parse model output:\n{raw}")
        return None

    parsed_set = {
        int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()
    }

    return parsed_set


DEFAULT_PARSER: parser_type = _parse_output


class MatcherBase:
    matcher_name: str
    max_retries: int
    window_size: int

    def __init__(self, **kwargs) -> None:
        # match_key -> list over recall idx[ list over attempts [ (prompt, response) ] ]
        self.prompt_response_log: dict[
            str, list[list[tuple[str, str, set[int] | None]]]
        ] = defaultdict(list)

    def _log_prompt_response(
        self,
        match_key: str,
        idx_recall: int,
        prompt: str,
        response: str,
        parsed_response: set[int] | None,
    ) -> None:
        while len(self.prompt_response_log[match_key]) <= idx_recall:
            self.prompt_response_log[match_key].append(list())
        self.prompt_response_log[match_key][idx_recall].append(
            (prompt, response, parsed_response)
        )

    def get_usage(self) -> dict | None:
        return None

    def _estimate_tokens(self, query: str) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def _calculate_cost(self, in_tokens: int, out_tokens: int) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def build_model_prompt(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        target_idx: int,
        prompt: str | None = None,
    ) -> str:
        """Return the full text prompt for one target recall segment.

        Parameters
        ----------
        story_segments : list of str
            All story segments of the story
        recall_segments : list of str
            All recall segments of the recall
        target_idx : int
            The index of the recall_segment to be scored in
            the recall_segments list. The recall window includes
            ``self.window_size`` segments on each side of it.
        prompt : str | None
            The 'static' prompt template. Can have following 'slots':
            {story_full}
            {story_segments}
            {recall_window}
            Only these slots are filled; any other braces are left as-is.
            If None, it uses DEFAULT_PROMPT.
        """

        # Steps 1 - 3 prepare dynamic components

        # 1. format story segments
        story_segments_str = format_story_segments(story_segments)

        # 2. build recall window
        window_str = build_recall_window(
            recall_segments,
            target_idx,
            self.window_size,
            start_marker=">>> ",
            end_marker=" <<<",
        )

        # 3. full story
        story_full = " ".join(seg.strip() for seg in story_segments)

        # Construct model prompt
        model_prompt = prompt or DEFAULT_PROMPT
        for slot, value in (
            ("{story_full}", story_full),
            ("{story_segments}", story_segments_str),
            ("{recall_window}", window_str),
        ):
            model_prompt = model_prompt.replace(slot, value)
        return model_prompt

    def match(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        match_key: str | None = None,
        prompt: str | None = None,
        parser: parser_type | None = None,
    ) -> matchlist_type:
        """Return each recall segment with its referenced story segments.

        Implementations should call ``_log_prompt_response`` after each model
        interaction (prompt and raw response text) for debugging and optional export.

        Parameters
        ----------
        story_segments: list[str]
            list of story segments
        recall_segments: list[str]
            list of recall segments
        match_key: str
            a string to identify story & recall segments. Used to save prompts and
            Responses to the prompt_response_log.
        prompt: str | None
            Prompt template, see ``build_model_prompt``. Defaults to DEFAULT_PROMPT.
        parser: parser_type | None
            Function extracting the set of (1-based) story segment numbers from the
            raw model output, or None if parsing fails (triggers a retry).
            Defaults to DEFAULT_PARSER.

        Returns
        -------
        match_list: list[tuple[int, list[int]]]
            list of matches:
            [
                (recall_segment_id_1, [story_segment_id_x, story_segment_id_y, ..]),
                (recall_segment_id_2, [story_segment_id_z, ...]),
                ...
            ]
        """
        raise NotImplementedError("Subclasses must implement this method")
