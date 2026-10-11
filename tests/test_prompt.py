"""Tests for prompt building and parser logic of the matchers."""

import pytest

from rmatch.matchers.matcher_base import DEFAULT_PARSER, MatcherBase
from rmatch.utils import build_recall_window, format_story_segments
from tests.conftest import make_anthropic_response

STORY_SEGMENTS = [
    "Alice met the Queen of Hearts.",
    "They sat down for tea.",
    "The Queen smiled and left.",
]

RECALL_SEGMENTS = [
    "Alice and the Queen drank tea.",
    "The Queen eventually left.",
]


def _build_prompt(target_idx: int, prompt: str | None = None) -> str:
    matcher = MatcherBase()
    matcher.window_size = 5
    return matcher.build_model_prompt(
        STORY_SEGMENTS, RECALL_SEGMENTS, target_idx, prompt=prompt
    )


# ── Parser correctness ────────────────────────────────────────────────────────


def test_parser_single_index():
    parser = DEFAULT_PARSER
    assert parser("<2>") == {2}


def test_parser_multiple_indices():
    parser = DEFAULT_PARSER
    assert parser("<1, 3>") == {1, 3}


def test_parser_none_response_returns_empty_set():
    parser = DEFAULT_PARSER
    result = parser("<NONE>")
    assert result == set()


def test_parser_malformed_response_returns_none():
    parser = DEFAULT_PARSER
    assert parser("I cannot determine the answer from the given text.") is None


def test_parser_empty_string_returns_none():
    parser = DEFAULT_PARSER
    assert parser("") is None


def test_parser_strips_surrounding_whitespace():
    parser = DEFAULT_PARSER
    result = parser("  <2>  ")
    assert result == {2}


def test_parser_handles_extra_whitespace_inside_brackets():
    parser = DEFAULT_PARSER
    result = parser("<1,  2,  3>")
    assert result == {1, 2, 3}


def test_parser_returns_none_for_text_without_angle_brackets():
    parser = DEFAULT_PARSER
    assert parser("The answer is segment 2.") is None


def test_parser_finds_match_in_longer_text():
    parser = DEFAULT_PARSER
    result = parser("After careful consideration, I think <1, 2> match.")
    assert result == {1, 2}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<0>", {0}),
        ("<NONE> <2>", set()),
        ("<1> and <2>", {1}),
        ("<>", None),
        ("<   >", set()),
        ("<1 2 3>", set()),
        ("<1, abc, 2>", None),
        ("<1, 1, 2>", {1, 2}),
        ("<01, 02>", {1, 2}),
    ],
)
def test_parser_edge_cases(raw, expected):
    parser = DEFAULT_PARSER
    assert parser(raw) == expected


# ── Prompt content ────────────────────────────────────────────────────────────


def test_prompt_contains_all_story_segments():
    prompt = _build_prompt(0)
    for seg in STORY_SEGMENTS:
        assert seg in prompt


def test_prompt_contains_target_recall_segment():
    prompt = _build_prompt(0)
    assert RECALL_SEGMENTS[0] in prompt


def test_prompt_contains_target_marker():
    prompt = _build_prompt(0)
    # Primary prompt uses >>> <<< markers
    assert ">>>" in prompt


def test_prompt_second_target_index():
    prompt = _build_prompt(1)
    assert RECALL_SEGMENTS[1] in prompt


# ── Custom prompt templates ───────────────────────────────────────────────────


def test_default_prompt_used_when_none():
    assert _build_prompt(0).startswith("This is the original story:")


def test_custom_prompt_fills_all_slots():
    template = "S:{story_full}|E:{story_segments}|R:{recall_window}"
    prompt = _build_prompt(1, prompt=template)
    assert "{" not in prompt
    assert prompt.startswith("S:" + " ".join(STORY_SEGMENTS))
    assert "1. Alice met the Queen of Hearts." in prompt
    assert f">>> {RECALL_SEGMENTS[1]} <<<" in prompt


def test_custom_prompt_keeps_literal_braces():
    template = 'Story: {story_segments}\nAnswer as JSON: {"matches": [1, 2]} {unknown}'
    prompt = _build_prompt(0, prompt=template)
    assert '{"matches": [1, 2]} {unknown}' in prompt
    assert "{story_segments}" not in prompt


def test_custom_parser_passed_to_match(anthropic_matcher, mock_anthropic_client):
    mock_anthropic_client.messages.create.return_value = make_anthropic_response(
        "ANSWER: 3"
    )

    def parser(raw: str) -> set[int] | None:
        return {int(raw.split(":")[1])}

    result = anthropic_matcher.match(
        STORY_SEGMENTS, ["recall"], prompt="{recall_window}", parser=parser
    )
    assert result == [(0, [2])]
    sent = mock_anthropic_client.messages.create.call_args.kwargs["messages"][0]
    assert sent["content"] == ">>> recall <<<"


# ── build_recall_window ───────────────────────────────────────────────────────


def test_build_recall_window_contains_target_marker():
    result = build_recall_window(RECALL_SEGMENTS, target_idx=0, window_size=5)
    assert "<target>" in result
    assert RECALL_SEGMENTS[0] in result


def test_build_recall_window_zero_window_only_target():
    result = build_recall_window(RECALL_SEGMENTS, target_idx=0, window_size=0)
    assert "<target>" in result
    assert RECALL_SEGMENTS[0] in result
    # With window_size=0, only the target segment should be present
    assert RECALL_SEGMENTS[1] not in result


def test_build_recall_window_includes_neighbors():
    segs = ["alpha_seg", "beta_seg", "gamma_seg", "delta_seg", "epsilon_seg"]
    result = build_recall_window(segs, target_idx=2, window_size=1)
    assert "beta_seg" in result
    assert "gamma_seg" in result
    assert "delta_seg" in result
    # Outside the window
    assert "alpha_seg" not in result
    assert "epsilon_seg" not in result


def test_build_recall_window_target_at_start():
    result = build_recall_window(RECALL_SEGMENTS, target_idx=0, window_size=2)
    assert "<target>" in result
    assert RECALL_SEGMENTS[0] in result


def test_build_recall_window_custom_markers():
    result = build_recall_window(
        RECALL_SEGMENTS,
        target_idx=0,
        window_size=0,
        start_marker=">>> ",
        end_marker=" <<<",
    )
    assert ">>>" in result
    assert "<<<" in result


# ── format_story_segments ─────────────────────────────────────────────────────


def test_format_story_segments_numbering():
    result = format_story_segments(["A", "B", "C"])
    assert "1. A" in result
    assert "2. B" in result
    assert "3. C" in result


def test_format_story_segments_single():
    result = format_story_segments(["Only segment"])
    assert "1. Only segment" in result


def test_format_story_segments_preserves_content():
    segs = STORY_SEGMENTS
    result = format_story_segments(segs)
    for seg in segs:
        assert seg.strip() in result
