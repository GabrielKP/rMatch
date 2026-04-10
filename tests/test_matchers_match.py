"""Tests for the match() method of all three LLM matchers.

All external calls (Anthropic API, OpenAI API, HuggingFace pipeline) are mocked.
The tests verify output format, correct index handling (1-based parser → 0-based
output), retry logic, and graceful degradation when responses are always malformed.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from tests.conftest import (
    RECALL_SEGMENTS,
    STORY_SEGMENTS,
    make_anthropic_response,
    make_openai_response,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _is_valid_matchlist(result) -> bool:
    """Check the canonical matchlist_type shape."""
    if not isinstance(result, list):
        return False
    return all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], int)
        and isinstance(item[1], list)
        and all(isinstance(x, int) for x in item[1])
        for item in result
    )


# ── MatcherAnthropic ──────────────────────────────────────────────────────────


class TestMatcherAnthropicMatch:
    def test_returns_correct_type(self, anthropic_matcher, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # Parser output "<2>" → 1-based index 2 → 0-based index 1
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<2>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["one recall"], max_retries=3)
        assert result == [(0, [1])]

    def test_multi_recall_valid_responses(
        self, anthropic_matcher, mock_anthropic_client
    ):
        mock_anthropic_client.messages.create.side_effect = [
            make_anthropic_response("<1>"),  # recall 0 → story seg 0
            make_anthropic_response("<2, 3>"),  # recall 1 → story segs 1 and 2
        ]
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        assert result == [(0, [0]), (1, [1, 2])]

    def test_none_response_returns_empty_story_indices(
        self, anthropic_matcher, mock_anthropic_client
    ):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<NONE>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall seg"], max_retries=3)
        assert result == [(0, [])]

    def test_malformed_response_triggers_retry(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # Two bad responses, then one good
        mock_anthropic_client.messages.create.side_effect = [
            make_anthropic_response("I cannot determine this."),
            make_anthropic_response("No match found."),
            make_anthropic_response("<1>"),
        ]
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert mock_anthropic_client.messages.create.call_count == 3
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # Always malformed → match should return empty list for that segment gracefully
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "could not parse"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert mock_anthropic_client.messages.create.call_count == 3
        assert result == [(0, [])]  # Fails gracefully, no crash

    def test_empty_recall_segments_returns_empty_list(
        self, anthropic_matcher, mock_anthropic_client
    ):
        result = anthropic_matcher.match(STORY_SEGMENTS, [], max_retries=3)
        assert result == []
        mock_anthropic_client.messages.create.assert_not_called()

    def test_single_story_segment(self, anthropic_matcher, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>"
        )
        result = anthropic_matcher.match(["Only segment"], ["recall"], max_retries=3)
        assert result == [(0, [0])]

    def test_output_indices_are_sorted(self, anthropic_matcher, mock_anthropic_client):
        # Parser returns {3, 1} (unordered) → should be stored sorted
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<3, 1>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(
        self, anthropic_matcher, mock_anthropic_client
    ):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>"
        )
        anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        # One log entry per recall segment (successful on first try)
        assert len(anthropic_matcher.prompt_response_log) == len(RECALL_SEGMENTS)

    def test_usage_metrics_updated(self, anthropic_matcher, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>", in_tokens=100, out_tokens=20
        )
        anthropic_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        usage = anthropic_matcher.get_usage()
        assert usage["in_tokens"] == 100
        assert usage["out_tokens"] == 20
        assert usage["cost"] > 0

    def test_recall_index_in_output_matches_position(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # Each item (i, ...) should have i = position in recall_segments
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        for expected_idx, (actual_idx, _) in enumerate(result):
            assert actual_idx == expected_idx


# ── MatcherOpenAI ─────────────────────────────────────────────────────────────


class TestMatcherOpenAIMatch:
    def test_returns_correct_type(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<1>")
        result = openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<2>")
        result = openai_matcher.match(STORY_SEGMENTS, ["one recall"], max_retries=3)
        assert result == [(0, [1])]

    def test_multi_recall_valid_responses(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.side_effect = [
            make_openai_response("<3>"),
            make_openai_response("<1, 2>"),
        ]
        result = openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        assert result == [(0, [2]), (1, [0, 1])]

    def test_none_response_returns_empty_story_indices(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "<NONE>"
        )
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert result == [(0, [])]

    def test_malformed_response_triggers_retry(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.side_effect = [
            make_openai_response("No idea."),
            make_openai_response("<1>"),
        ]
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert mock_openai_client.responses.create.call_count == 2
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "unparseable"
        )
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert mock_openai_client.responses.create.call_count == 3
        assert result == [(0, [])]

    def test_empty_recall_segments_returns_empty_list(
        self, openai_matcher, mock_openai_client
    ):
        result = openai_matcher.match(STORY_SEGMENTS, [], max_retries=3)
        assert result == []
        mock_openai_client.responses.create.assert_not_called()

    def test_output_indices_sorted(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "<3, 1, 2>"
        )
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<1>")
        openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        assert len(openai_matcher.prompt_response_log) == len(RECALL_SEGMENTS)


# ── MatcherHuggingFace ────────────────────────────────────────────────────────


class TestMatcherHuggingFaceMatch:
    def _set_pipe_response(self, mock_hf_pipe, texts: list[str]):
        """Configure pipe to return one response per prompt."""
        mock_hf_pipe.return_value = [[{"generated_text": t}] for t in texts]

    def test_returns_correct_type(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<2>"])
        result = huggingface_matcher.match(
            STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3
        )
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<2>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert result == [(0, [1])]

    def test_multi_recall_correct_output(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<3>"])
        result = huggingface_matcher.match(
            STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3
        )
        assert result == [(0, [0]), (1, [2])]

    def test_none_response_returns_empty_story_indices(
        self, huggingface_matcher, mock_hf_pipe
    ):
        self._set_pipe_response(mock_hf_pipe, ["<NONE>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert result == [(0, [])]

    def test_batches_all_segments_in_first_call(
        self, huggingface_matcher, mock_hf_pipe
    ):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<2>"])
        huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        # Pipeline is called once with both prompts in a single batch
        assert mock_hf_pipe.call_count == 1
        first_call_input = mock_hf_pipe.call_args[0][0]
        assert len(first_call_input) == len(RECALL_SEGMENTS)

    def test_malformed_response_triggers_retry(self, huggingface_matcher, mock_hf_pipe):
        # First call: malformed. Second call: valid.
        mock_hf_pipe.side_effect = [
            [[{"generated_text": "could not parse"}]],
            [[{"generated_text": "<1>"}]],
        ]
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert mock_hf_pipe.call_count == 2
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(
        self, huggingface_matcher, mock_hf_pipe
    ):
        mock_hf_pipe.return_value = [[{"generated_text": "unparseable"}]]
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert mock_hf_pipe.call_count == 3
        assert result == [(0, [])]  # Fails gracefully, no crash

    def test_empty_recall_returns_empty_list(self, huggingface_matcher, mock_hf_pipe):
        result = huggingface_matcher.match(STORY_SEGMENTS, [], max_retries=3)
        assert result == []
        mock_hf_pipe.assert_not_called()

    def test_out_of_bounds_index_is_treated_as_malformed(
        self, huggingface_matcher, mock_hf_pipe
    ):
        # Index 999 is beyond len(STORY_SEGMENTS)=3; treated as invalid → retry
        mock_hf_pipe.side_effect = [
            [[{"generated_text": "<999>"}]],  # Out of bounds
            [[{"generated_text": "<1>"}]],  # Valid on retry
        ]
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        assert result == [(0, [0])]

    def test_partial_batch_retry(self, huggingface_matcher, mock_hf_pipe):
        # Two recalls: first is valid, second is malformed on first call.
        mock_hf_pipe.side_effect = [
            # First batch call: recall0=valid, recall1=malformed
            [[{"generated_text": "<1>"}], [{"generated_text": "bad"}]],
            # Retry batch (only recall1): valid
            [[{"generated_text": "<2>"}]],
        ]
        result = huggingface_matcher.match(
            STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3
        )
        assert result == [(0, [0]), (1, [1])]
        # Pipeline called twice: full batch + single retry
        assert mock_hf_pipe.call_count == 2

    def test_output_indices_sorted(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<3, 1>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"], max_retries=3)
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<2>"])
        huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, max_retries=3)
        assert len(huggingface_matcher.prompt_response_log) == len(RECALL_SEGMENTS)
