"""Tests for the match() method of all three LLM matchers.

All external calls (Anthropic API, OpenAI API, HuggingFace pipeline) are mocked.
The tests verify output format, correct index handling (1-based parser → 0-based
output), retry logic, and graceful degradation when responses are always malformed.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import (
    RECALL_SEGMENTS,
    STORY_SEGMENTS,
    make_anthropic_response,
    make_mlx_response,
    make_openai_response,
    make_vllm_output,
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
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # Parser output "<2>" → 1-based index 2 → 0-based index 1
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<2>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["one recall"])
        assert result == [(0, [1])]

    def test_multi_recall_valid_responses(
        self, anthropic_matcher, mock_anthropic_client
    ):
        mock_anthropic_client.messages.create.side_effect = [
            make_anthropic_response("<1>"),  # recall 0 → story seg 0
            make_anthropic_response("<2, 3>"),  # recall 1 → story segs 1 and 2
        ]
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [0]), (1, [1, 2])]

    def test_none_response_returns_empty_story_indices(
        self, anthropic_matcher, mock_anthropic_client
    ):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<NONE>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall seg"])
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
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_anthropic_client.messages.create.call_count == 3
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # Always malformed → match should return empty list for that segment gracefully
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "could not parse"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_anthropic_client.messages.create.call_count == 3
        assert result == [(0, [])]  # Fails gracefully, no crash

    def test_empty_recall_segments_returns_empty_list(
        self, anthropic_matcher, mock_anthropic_client
    ):
        result = anthropic_matcher.match(STORY_SEGMENTS, [])
        assert result == []
        mock_anthropic_client.messages.create.assert_not_called()

    def test_single_story_segment(self, anthropic_matcher, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>"
        )
        result = anthropic_matcher.match(["Only segment"], ["recall"])
        assert result == [(0, [0])]

    def test_output_indices_are_sorted(self, anthropic_matcher, mock_anthropic_client):
        # Parser returns {3, 1} (unordered) → should be stored sorted
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<3, 1>"
        )
        result = anthropic_matcher.match(STORY_SEGMENTS, ["recall"])
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(
        self, anthropic_matcher, mock_anthropic_client
    ):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>"
        )
        anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, "story1_sub-01_1")
        # One log entry per recall segment (successful on first try)
        assert len(anthropic_matcher.prompt_response_log["story1_sub-01_1"]) == len(
            RECALL_SEGMENTS
        )

    def test_usage_metrics_updated(self, anthropic_matcher, mock_anthropic_client):
        mock_anthropic_client.messages.create.return_value = make_anthropic_response(
            "<1>", in_tokens=100, out_tokens=20
        )
        anthropic_matcher.match(STORY_SEGMENTS, ["recall"])
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
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        for expected_idx, (actual_idx, _) in enumerate(result):
            assert actual_idx == expected_idx

    def test_prompt_response_log_content(
        self, anthropic_matcher, mock_anthropic_client
    ):
        # recall 0: attempt 1 malformed → attempt 2 succeeds with "<2>" → parsed {2}
        # recall 1: attempt 1 succeeds with "<1>" → parsed {1}
        mock_anthropic_client.messages.create.side_effect = [
            make_anthropic_response("bad response"),
            make_anthropic_response("<2>"),
            make_anthropic_response("<1>"),
        ]
        key = "story1_sub-01_1"
        result = anthropic_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, key)

        log = anthropic_matcher.prompt_response_log[key]

        # recall 0: two attempts logged
        assert len(log[0]) == 2
        prompt0_a1, response0_a1, parsed0_a1 = log[0][0]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a1
        assert response0_a1 == "bad response"
        assert parsed0_a1 is None
        prompt0_a2, response0_a2, parsed0_a2 = log[0][1]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a2
        assert response0_a2 == "<2>"
        assert parsed0_a2 == {2}

        # recall 1: one attempt logged
        assert len(log[1]) == 1
        prompt1, response1, parsed1 = log[1][0]
        assert f">>> {RECALL_SEGMENTS[1]}" in prompt1
        assert response1 == "<1>"
        assert parsed1 == {1}

        # final parsed_response in the log maps to the returned 0-based story indices
        _, story_indices_0 = result[0]
        assert sorted(x - 1 for x in parsed0_a2) == story_indices_0
        _, story_indices_1 = result[1]
        assert sorted(x - 1 for x in parsed1) == story_indices_1


# ── MatcherOpenAI ─────────────────────────────────────────────────────────────


class TestMatcherOpenAIMatch:
    def test_returns_correct_type(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<1>")
        result = openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<2>")
        result = openai_matcher.match(STORY_SEGMENTS, ["one recall"])
        assert result == [(0, [1])]

    def test_multi_recall_valid_responses(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.side_effect = [
            make_openai_response("<3>"),
            make_openai_response("<1, 2>"),
        ]
        result = openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [2]), (1, [0, 1])]

    def test_none_response_returns_empty_story_indices(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "<NONE>"
        )
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [])]

    def test_malformed_response_triggers_retry(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.side_effect = [
            make_openai_response("No idea."),
            make_openai_response("<1>"),
        ]
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_openai_client.responses.create.call_count == 2
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "unparseable"
        )
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_openai_client.responses.create.call_count == 3
        assert result == [(0, [])]

    def test_empty_recall_segments_returns_empty_list(
        self, openai_matcher, mock_openai_client
    ):
        result = openai_matcher.match(STORY_SEGMENTS, [])
        assert result == []
        mock_openai_client.responses.create.assert_not_called()

    def test_output_indices_sorted(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "<3, 1, 2>"
        )
        result = openai_matcher.match(STORY_SEGMENTS, ["recall"])
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<1>")
        openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, "story1_sub-01_1")
        assert len(openai_matcher.prompt_response_log["story1_sub-01_1"]) == len(
            RECALL_SEGMENTS
        )

    def test_single_story_segment(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response("<1>")
        result = openai_matcher.match(["Only segment"], ["recall"])
        assert result == [(0, [0])]

    def test_usage_metrics_updated(self, openai_matcher, mock_openai_client):
        mock_openai_client.responses.create.return_value = make_openai_response(
            "<1>", in_tokens=100, out_tokens=20
        )
        openai_matcher.match(STORY_SEGMENTS, ["recall"])
        usage = openai_matcher.get_usage()
        assert usage["in_tokens"] == 100
        assert usage["out_tokens"] == 20
        assert usage["cost"] > 0

    def test_recall_index_in_output_matches_position(
        self, openai_matcher, mock_openai_client
    ):
        mock_openai_client.responses.create.return_value = make_openai_response("<1>")
        result = openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        for expected_idx, (actual_idx, _) in enumerate(result):
            assert actual_idx == expected_idx

    def test_prompt_response_log_content(self, openai_matcher, mock_openai_client):
        # recall 0: attempt 1 malformed → attempt 2 succeeds with "<3>" → parsed {3}
        # recall 1: attempt 1 succeeds with "<1>" → parsed {1}
        mock_openai_client.responses.create.side_effect = [
            make_openai_response("bad response"),
            make_openai_response("<3>"),
            make_openai_response("<1>"),
        ]
        key = "story1_sub-01_1"
        result = openai_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, key)

        log = openai_matcher.prompt_response_log[key]

        # recall 0: two attempts logged
        assert len(log[0]) == 2
        prompt0_a1, response0_a1, parsed0_a1 = log[0][0]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a1
        assert response0_a1 == "bad response"
        assert parsed0_a1 is None
        prompt0_a2, response0_a2, parsed0_a2 = log[0][1]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a2
        assert response0_a2 == "<3>"
        assert parsed0_a2 == {3}

        # recall 1: one attempt logged
        assert len(log[1]) == 1
        prompt1, response1, parsed1 = log[1][0]
        assert f">>> {RECALL_SEGMENTS[1]}" in prompt1
        assert response1 == "<1>"
        assert parsed1 == {1}

        # final parsed_response in the log maps to the returned 0-based story indices
        _, story_indices_0 = result[0]
        assert sorted(x - 1 for x in parsed0_a2) == story_indices_0
        _, story_indices_1 = result[1]
        assert sorted(x - 1 for x in parsed1) == story_indices_1


# ── MatcherHuggingFace ────────────────────────────────────────────────────────


class TestMatcherHuggingFaceMatch:
    def _set_pipe_response(self, mock_hf_pipe, texts: list[str]):
        """Configure pipe to return one response per prompt."""
        mock_hf_pipe.return_value = [[{"generated_text": t}] for t in texts]

    def test_returns_correct_type(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<2>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<2>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [1])]

    def test_multi_recall_correct_output(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<3>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [0]), (1, [2])]

    def test_none_response_returns_empty_story_indices(
        self, huggingface_matcher, mock_hf_pipe
    ):
        self._set_pipe_response(mock_hf_pipe, ["<NONE>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [])]

    def test_batches_all_segments_in_first_call(
        self, huggingface_matcher, mock_hf_pipe
    ):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<2>"])
        huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
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
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_hf_pipe.call_count == 2
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(
        self, huggingface_matcher, mock_hf_pipe
    ):
        mock_hf_pipe.return_value = [[{"generated_text": "unparseable"}]]
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_hf_pipe.call_count == 3
        assert result == [(0, [])]  # Fails gracefully, no crash

    def test_empty_recall_returns_empty_list(self, huggingface_matcher, mock_hf_pipe):
        result = huggingface_matcher.match(STORY_SEGMENTS, [])
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
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [0])]

    def test_partial_batch_retry(self, huggingface_matcher, mock_hf_pipe):
        # Two recalls: first is valid, second is malformed on first call.
        mock_hf_pipe.side_effect = [
            # First batch call: recall0=valid, recall1=malformed
            [[{"generated_text": "<1>"}], [{"generated_text": "bad"}]],
            # Retry batch (only recall1): valid
            [[{"generated_text": "<2>"}]],
        ]
        result = huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [0]), (1, [1])]
        # Pipeline called twice: full batch + single retry
        assert mock_hf_pipe.call_count == 2

    def test_output_indices_sorted(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<3, 1>"])
        result = huggingface_matcher.match(STORY_SEGMENTS, ["recall"])
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(self, huggingface_matcher, mock_hf_pipe):
        self._set_pipe_response(mock_hf_pipe, ["<1>", "<2>"])
        huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, "story1_sub-01_1")
        assert len(huggingface_matcher.prompt_response_log["story1_sub-01_1"]) == len(
            RECALL_SEGMENTS
        )

    def test_prompt_response_log_content(self, huggingface_matcher, mock_hf_pipe):
        # recall 0: attempt 1 malformed → retried alone in attempt 2 with "<2>"
        # recall 1: attempt 1 succeeds with "<1>" → parsed {1}
        mock_hf_pipe.side_effect = [
            # First batch: both recalls together; recall 0 malformed, recall 1 valid
            [[{"generated_text": "bad response"}], [{"generated_text": "<1>"}]],
            # Retry batch (only recall 0 pending)
            [[{"generated_text": "<2>"}]],
        ]
        key = "story1_sub-01_1"
        result = huggingface_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, key)

        log = huggingface_matcher.prompt_response_log[key]

        # recall 0: two attempts logged
        assert len(log[0]) == 2
        prompt0_a1, response0_a1, parsed0_a1 = log[0][0]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a1
        assert response0_a1 == "bad response"
        assert parsed0_a1 is None
        prompt0_a2, response0_a2, parsed0_a2 = log[0][1]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a2
        assert response0_a2 == "<2>"
        assert parsed0_a2 == {2}

        # recall 1: one attempt logged
        assert len(log[1]) == 1
        prompt1, response1, parsed1 = log[1][0]
        assert f">>> {RECALL_SEGMENTS[1]}" in prompt1
        assert response1 == "<1>"
        assert parsed1 == {1}

        # final parsed_response in the log maps to the returned 0-based story indices
        _, story_indices_0 = result[0]
        assert sorted(x - 1 for x in parsed0_a2) == story_indices_0
        _, story_indices_1 = result[1]
        assert sorted(x - 1 for x in parsed1) == story_indices_1


# ── MatcherCuda ───────────────────────────────────────────────────────────────


class TestMatcherCudaMatch:
    def _set_generate_response(self, mock_vllm_llm, texts: list[str]):
        """Configure llm.generate to return one output per prompt."""
        mock_vllm_llm.generate.return_value = [make_vllm_output(t) for t in texts]

    def test_returns_correct_type(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<1>", "<2>"])
        result = vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<2>"])
        result = vllm_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [1])]

    def test_multi_recall_correct_output(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<1>", "<3>"])
        result = vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [0]), (1, [2])]

    def test_none_response_returns_empty_story_indices(
        self, vllm_matcher, mock_vllm_llm
    ):
        self._set_generate_response(mock_vllm_llm, ["<NONE>"])
        result = vllm_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [])]

    def test_batches_all_segments_in_first_call(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<1>", "<2>"])
        vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert mock_vllm_llm.generate.call_count == 1
        first_call_input = mock_vllm_llm.generate.call_args[0][0]
        assert len(first_call_input) == len(RECALL_SEGMENTS)

    def test_malformed_response_triggers_retry(self, vllm_matcher, mock_vllm_llm):
        mock_vllm_llm.generate.side_effect = [
            [make_vllm_output("could not parse")],
            [make_vllm_output("<1>")],
        ]
        result = vllm_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_vllm_llm.generate.call_count == 2
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(self, vllm_matcher, mock_vllm_llm):
        mock_vllm_llm.generate.return_value = [make_vllm_output("unparseable")]
        result = vllm_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_vllm_llm.generate.call_count == 3
        assert result == [(0, [])]

    def test_empty_recall_returns_empty_list(self, vllm_matcher, mock_vllm_llm):
        result = vllm_matcher.match(STORY_SEGMENTS, [])
        assert result == []
        mock_vllm_llm.generate.assert_not_called()

    def test_out_of_bounds_index_is_treated_as_malformed(
        self, vllm_matcher, mock_vllm_llm
    ):
        mock_vllm_llm.generate.side_effect = [
            [make_vllm_output("<999>")],
            [make_vllm_output("<1>")],
        ]
        result = vllm_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [0])]

    def test_partial_batch_retry(self, vllm_matcher, mock_vllm_llm):
        mock_vllm_llm.generate.side_effect = [
            [make_vllm_output("<1>"), make_vllm_output("bad")],
            [make_vllm_output("<2>")],
        ]
        result = vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [0]), (1, [1])]
        assert mock_vllm_llm.generate.call_count == 2

    def test_output_indices_sorted(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<3, 1>"])
        result = vllm_matcher.match(STORY_SEGMENTS, ["recall"])
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_prompt_response_log_populated(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<1>", "<2>"])
        vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, "story1_sub-01_1")
        assert len(vllm_matcher.prompt_response_log["story1_sub-01_1"]) == len(
            RECALL_SEGMENTS
        )

    def test_recall_index_in_output_matches_position(self, vllm_matcher, mock_vllm_llm):
        self._set_generate_response(mock_vllm_llm, ["<1>", "<2>"])
        result = vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        for expected_idx, (actual_idx, _) in enumerate(result):
            assert actual_idx == expected_idx

    def test_prompt_response_log_content(self, vllm_matcher, mock_vllm_llm):
        # recall 0: attempt 1 malformed
        # ->retried alone with "<2>"; recall 1: succeeds "<1>"
        mock_vllm_llm.generate.side_effect = [
            [make_vllm_output("bad response"), make_vllm_output("<1>")],
            [make_vllm_output("<2>")],
        ]
        key = "story1_sub-01_1"
        result = vllm_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, key)

        log = vllm_matcher.prompt_response_log[key]

        assert len(log[0]) == 2
        prompt0_a1, response0_a1, parsed0_a1 = log[0][0]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a1
        assert response0_a1 == "bad response"
        assert parsed0_a1 is None

        prompt0_a2, response0_a2, parsed0_a2 = log[0][1]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a2
        assert response0_a2 == "<2>"
        assert parsed0_a2 == {2}

        assert len(log[1]) == 1
        prompt1, response1, parsed1 = log[1][0]
        assert f">>> {RECALL_SEGMENTS[1]}" in prompt1
        assert response1 == "<1>"
        assert parsed1 == {1}

        _, story_indices_0 = result[0]
        assert sorted(x - 1 for x in parsed0_a2) == story_indices_0
        _, story_indices_1 = result[1]
        assert sorted(x - 1 for x in parsed1) == story_indices_1


# ── MatcherMac ────────────────────────────────────────────────────────────────


class TestMatcherMacMatch:
    def test_returns_correct_type(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.side_effect = [
            make_mlx_response("<1>"),
            make_mlx_response("<2>"),
        ]
        result = mlx_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert _is_valid_matchlist(result)

    def test_single_recall_valid_response(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.return_value = make_mlx_response("<2>")
        result = mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [1])]

    def test_multi_recall_correct_output(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.side_effect = [
            make_mlx_response("<1>"),
            make_mlx_response("<3>"),
        ]
        result = mlx_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert result == [(0, [0]), (1, [2])]

    def test_none_response_returns_empty_story_indices(
        self, mlx_matcher, mock_mlx_generate
    ):
        mock_mlx_generate.return_value = make_mlx_response("<NONE>")
        result = mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [])]

    def test_processes_segments_sequentially(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.side_effect = [
            make_mlx_response("<1>"),
            make_mlx_response("<2>"),
        ]
        mlx_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        assert mock_mlx_generate.call_count == len(RECALL_SEGMENTS)

    def test_malformed_response_triggers_retry(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.side_effect = [
            make_mlx_response("could not parse"),
            make_mlx_response("<1>"),
        ]
        result = mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_mlx_generate.call_count == 2
        assert result == [(0, [0])]

    def test_all_retries_exhausted_returns_empty(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.return_value = make_mlx_response("unparseable")
        result = mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        assert mock_mlx_generate.call_count == 3
        assert result == [(0, [])]

    def test_empty_recall_returns_empty_list(self, mlx_matcher, mock_mlx_generate):
        result = mlx_matcher.match(STORY_SEGMENTS, [])
        assert result == []
        mock_mlx_generate.assert_not_called()

    def test_out_of_bounds_index_is_accepted(self, mlx_matcher, mock_mlx_generate):
        # MLX does not validate bounds; any parseable index is stored as-is
        mock_mlx_generate.return_value = make_mlx_response("<999>")
        result = mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        assert result == [(0, [998])]
        assert mock_mlx_generate.call_count == 1

    def test_output_indices_sorted(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.return_value = make_mlx_response("<3, 1>")
        result = mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        story_indices = result[0][1]
        assert story_indices == sorted(story_indices)

    def test_usage_metrics_updated(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.return_value = make_mlx_response(
            "<1>",
            prompt_tokens=50,
            generation_tokens=10,
            generation_tps=25.0,
            peak_memory=2.0,
        )
        mlx_matcher.match(STORY_SEGMENTS, ["recall"])
        usage = mlx_matcher.get_usage()
        assert usage["total_prompt_tokens"] == 50
        assert usage["total_generation_tokens"] == 10
        assert usage["avg_token_per_second"] == 25.0
        assert usage["peak_memory"] == 2.0

    def test_recall_index_in_output_matches_position(
        self, mlx_matcher, mock_mlx_generate
    ):
        mock_mlx_generate.side_effect = [
            make_mlx_response("<1>"),
            make_mlx_response("<2>"),
        ]
        result = mlx_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS)
        for expected_idx, (actual_idx, _) in enumerate(result):
            assert actual_idx == expected_idx

    def test_prompt_response_log_populated(self, mlx_matcher, mock_mlx_generate):
        mock_mlx_generate.side_effect = [
            make_mlx_response("<1>"),
            make_mlx_response("<2>"),
        ]
        mlx_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, "story1_sub-01_1")
        assert len(mlx_matcher.prompt_response_log["story1_sub-01_1"]) == len(
            RECALL_SEGMENTS
        )

    def test_prompt_response_log_content(self, mlx_matcher, mock_mlx_generate):
        # recall 0: attempt 1 malformed
        # -> attempt 2 succeeds "<2>"; recall 1: succeeds "<1>"
        mock_mlx_generate.side_effect = [
            make_mlx_response("bad response"),
            make_mlx_response("<2>"),
            make_mlx_response("<1>"),
        ]
        key = "story1_sub-01_1"
        result = mlx_matcher.match(STORY_SEGMENTS, RECALL_SEGMENTS, key)

        log = mlx_matcher.prompt_response_log[key]

        assert len(log[0]) == 2
        prompt0_a1, response0_a1, parsed0_a1 = log[0][0]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a1
        assert response0_a1 == "bad response"
        assert parsed0_a1 is None

        prompt0_a2, response0_a2, parsed0_a2 = log[0][1]
        assert f">>> {RECALL_SEGMENTS[0]}" in prompt0_a2
        assert response0_a2 == "<2>"
        assert parsed0_a2 == {2}

        assert len(log[1]) == 1
        prompt1, response1, parsed1 = log[1][0]
        assert f">>> {RECALL_SEGMENTS[1]}" in prompt1
        assert response1 == "<1>"
        assert parsed1 == {1}

        _, story_indices_0 = result[0]
        assert sorted(x - 1 for x in parsed0_a2) == story_indices_0
        _, story_indices_1 = result[1]
        assert sorted(x - 1 for x in parsed1) == story_indices_1


# ── max_retries init ─────────────────────────────────────────────────────────


class TestMaxRetriesInit:
    """max_retries is an __init__ parameter, not a match() parameter."""

    def test_anthropic_max_retries_stored_from_init(self, mock_anthropic_client):
        with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
            from rmatch.matchers.matcher_anthropic import MatcherAnthropic

            m = MatcherAnthropic(api_key="test-key", max_retries=5)
        assert m.max_retries == 5

    def test_openai_max_retries_stored_from_init(self, mock_openai_client):
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_openai_client
        with patch.dict("sys.modules", {"openai": mock_openai_mod}):
            from rmatch.matchers.matcher_openai import MatcherOpenAI

            m = MatcherOpenAI(api_key="test-key", max_retries=7)
        assert m.max_retries == 7

    def test_huggingface_max_retries_stored_from_init(self, mock_hf_pipe):
        sys.modules["transformers"].pipeline.return_value = mock_hf_pipe
        from rmatch.matchers.matcher_huggingface import MatcherHuggingFace

        m = MatcherHuggingFace(api_key="test-hf-token", max_retries=2)
        assert m.max_retries == 2

    def test_vllm_max_retries_stored_from_init(self):
        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value.get_tokenizer.return_value.pad_token_id = None
        mock_vllm.LLM.return_value.get_tokenizer.return_value.eos_token_id = 2
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            from rmatch.matchers.matcher_cuda import MatcherCuda

            m = MatcherCuda(model_name="test-model", max_retries=5)
        assert m.max_retries == 5

    def test_mlx_max_retries_stored_from_init(self):
        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (MagicMock(), MagicMock())
        with patch.dict(
            "sys.modules",
            {
                "mlx_vlm": mock_mlx,
                "mlx_vlm.utils": MagicMock(),
                "mlx_vlm.prompt_utils": MagicMock(),
            },
        ):
            from rmatch.matchers.matcher_mac import MatcherMac

            m = MatcherMac(model_name="test-model", max_retries=7)
        assert m.max_retries == 7

    def test_default_max_retries_when_not_specified(self, mock_anthropic_client):
        with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
            from rmatch.matchers.matcher_anthropic import MatcherAnthropic

            m = MatcherAnthropic(api_key="test-key")
        assert m.max_retries == 10


# ── Matcher factory ───────────────────────────────────────────────────────────


class TestMatcherFactory:
    """The public Python API entry point is Matcher(matcher_name=...).

    All matcher fixtures bypass the factory and instantiate subclasses directly,
    so breakage in Matcher.__new__ dispatch would go undetected without these tests.
    """

    def test_factory_returns_anthropic_instance(self):
        with patch("anthropic.Anthropic"):
            from rmatch.matchers import Matcher, MatcherAnthropic

            m = Matcher(matcher_name="anthropic", api_key="test-key")
        assert isinstance(m, MatcherAnthropic)

    def test_factory_returns_openai_instance(self):
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            from rmatch.matchers import Matcher, MatcherOpenAI

            m = Matcher(matcher_name="openai", api_key="test-key")
        assert isinstance(m, MatcherOpenAI)

    def test_factory_returns_huggingface_instance(self):
        pipe = MagicMock()
        pipe.tokenizer.pad_token_id = None
        pipe.tokenizer.eos_token_id = 2
        sys.modules["transformers"].pipeline.return_value = pipe
        from rmatch.matchers import Matcher, MatcherHuggingFace

        m = Matcher(matcher_name="huggingface", api_key="test-hf-token")
        assert isinstance(m, MatcherHuggingFace)

    def test_factory_returns_vllm_instance(self):
        mock_vllm = MagicMock()
        mock_vllm.LLM.return_value.get_tokenizer.return_value.pad_token_id = None
        mock_vllm.LLM.return_value.get_tokenizer.return_value.eos_token_id = 2
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            from rmatch.matchers import Matcher, MatcherCuda

            m = Matcher(matcher_name="cuda", model_name="test-model")
        assert isinstance(m, MatcherCuda)

    def test_factory_returns_mlx_instance(self):
        mock_mlx = MagicMock()
        mock_mlx.load.return_value = (MagicMock(), MagicMock())
        with patch.dict(
            "sys.modules",
            {
                "mlx_vlm": mock_mlx,
                "mlx_vlm.utils": MagicMock(),
                "mlx_vlm.prompt_utils": MagicMock(),
            },
        ):
            from rmatch.matchers import Matcher, MatcherMac

            m = Matcher(matcher_name="mac", model_name="test-model")
        assert isinstance(m, MatcherMac)

    def test_factory_unknown_name_raises_value_error(self):
        from rmatch.matchers import Matcher

        with pytest.raises(ValueError, match="Unknown matcher"):
            Matcher(matcher_name="nonexistent_backend")

    def test_factory_missing_name_raises_type_error(self):
        from rmatch.matchers import Matcher

        with pytest.raises(TypeError, match="matcher_name"):
            Matcher()
