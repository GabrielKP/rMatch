"""Tests for rmatch.prompt: prompt building and parser logic."""

import pytest

from rmatch.prompt import (
    build_recall_window,
    format_story_segments,
    get_prompt_and_parser,
)

STORY_SEGMENTS = [
    "Alice met the Queen of Hearts.",
    "They sat down for tea.",
    "The Queen smiled and left.",
]

RECALL_SEGMENTS = [
    "Alice and the Queen drank tea.",
    "The Queen eventually left.",
]


# ── Parser correctness ────────────────────────────────────────────────────────


def test_parser_single_index():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert parser("<2>") == {2}


def test_parser_multiple_indices():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert parser("<1, 3>") == {1, 3}


def test_parser_none_response_returns_empty_set():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    result = parser("<NONE>")
    assert result == set()


def test_parser_malformed_response_returns_none():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert parser("I cannot determine the answer from the given text.") is None


def test_parser_empty_string_returns_none():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert parser("") is None


def test_parser_strips_surrounding_whitespace():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    result = parser("  <2>  ")
    assert result == {2}


def test_parser_handles_extra_whitespace_inside_brackets():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    result = parser("<1,  2,  3>")
    assert result == {1, 2, 3}


def test_parser_returns_none_for_text_without_angle_brackets():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert parser("The answer is segment 2.") is None


def test_parser_finds_match_in_longer_text():
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
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
    _, parser = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert parser(raw) == expected


# ── Prompt content ────────────────────────────────────────────────────────────


def test_prompt_contains_all_story_segments():
    prompt, _ = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    for seg in STORY_SEGMENTS:
        assert seg in prompt


def test_prompt_contains_target_recall_segment():
    prompt, _ = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    assert RECALL_SEGMENTS[0] in prompt


def test_prompt_contains_target_marker():
    prompt, _ = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5)
    # Primary prompt uses >>> <<< markers
    assert ">>>" in prompt


def test_prompt_second_target_index():
    prompt, _ = get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 1, 5)
    assert RECALL_SEGMENTS[1] in prompt


# ── All prompt types ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prompt_type",
    [
        "primary",
        "primary_no_story",
        "primary_no_cot",
        "primary_no_story_no_cot",
        "secondary",
    ],
)
def test_all_prompt_types_return_str_and_callable(prompt_type):
    p, parser = get_prompt_and_parser(
        STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5, prompt=prompt_type
    )
    assert isinstance(p, str)
    assert len(p) > 0
    assert callable(parser)


def test_invalid_prompt_type_raises_value_error():
    with pytest.raises(ValueError, match="Invalid prompt type"):
        get_prompt_and_parser(STORY_SEGMENTS, RECALL_SEGMENTS, 0, 5, prompt="bogus")


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
