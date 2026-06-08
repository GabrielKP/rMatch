"""Tests for SegmenterCuda in rmatch.segmenters.

All vLLM / torch calls are mocked so the suite runs on CPU without any GPU
or model weights.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_vllm_output(text: str) -> MagicMock:
    """Build a fake vLLM RequestOutput with a single completion."""
    output = MagicMock()
    output.outputs[0].text = text
    return output


def _make_vllm_module() -> MagicMock:
    """Return a minimal fake vllm module."""
    mock_vllm = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 2
    mock_vllm.LLM.return_value.get_tokenizer.return_value = tokenizer
    mock_vllm.SamplingParams.return_value = MagicMock()
    return mock_vllm


def _make_torch_module() -> MagicMock:
    """Return a fake torch module where cuda.device_count() returns an int."""
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    return mock_torch


def _make_segmenter(granularity: str = "idea", max_retries: int = 3):
    """Instantiate SegmenterCuda with a fully mocked vLLM backend."""
    mock_vllm = _make_vllm_module()
    with patch.dict("sys.modules", {"vllm": mock_vllm, "torch": _make_torch_module()}):
        from rmatch.segmenters import SegmenterCuda

        seg = SegmenterCuda(
            model_name="test-model",
            max_retries=max_retries,
            granularity=granularity,
        )
    seg.llm = mock_vllm.LLM.return_value
    seg.tokenizer = mock_vllm.LLM.return_value.get_tokenizer.return_value
    # Chat template returns the user content verbatim so prompts are transparent
    seg.tokenizer.apply_chat_template.side_effect = lambda msgs, **kw: msgs[0][
        "content"
    ]
    return seg


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def segmenter():
    return _make_segmenter(granularity="idea")


@pytest.fixture()
def event_segmenter():
    return _make_segmenter(granularity="event")


# ── SegmenterCuda.__init__ ────────────────────────────────────────────────────


class TestSegmenterCudaInit:
    def test_default_model_name(self):
        mock_vllm = _make_vllm_module()

        with patch.dict(
            "sys.modules",
            {"vllm": mock_vllm, "torch": _make_torch_module()},
        ):
            from rmatch.segmenters import SegmenterCuda

            seg = SegmenterCuda()

        assert seg.model_name == "google/gemma-4-31B-it"

    def test_custom_model_name(self):
        mock_vllm = _make_vllm_module()
        with patch.dict(
            "sys.modules", {"vllm": mock_vllm, "torch": _make_torch_module()}
        ):
            from rmatch.segmenters import SegmenterCuda

            seg = SegmenterCuda(model_name="mistralai/Mistral-7B-Instruct-v0.3")
        assert seg.model_name == "mistralai/Mistral-7B-Instruct-v0.3"

    def test_max_new_tokens_default(self):
        assert _make_segmenter().max_new_tokens == 4096

    def test_max_retries_default(self):
        mock_vllm = _make_vllm_module()
        with patch.dict(
            "sys.modules", {"vllm": mock_vllm, "torch": _make_torch_module()}
        ):
            from rmatch.segmenters import SegmenterCuda

            seg = SegmenterCuda(model_name="x")
        assert seg.max_retries == 10

    def test_max_retries_custom(self):
        assert _make_segmenter(max_retries=5).max_retries == 5

    def test_granularity_idea_and_event_accepted(self):
        assert _make_segmenter(granularity="idea").granularity == "idea"
        assert _make_segmenter(granularity="event").granularity == "event"

    def test_invalid_granularity_raises(self):
        mock_vllm = _make_vllm_module()
        with patch.dict(
            "sys.modules", {"vllm": mock_vllm, "torch": _make_torch_module()}
        ):
            from rmatch.segmenters import SegmenterCuda

            with pytest.raises(ValueError, match="Granularity"):
                SegmenterCuda(model_name="x", granularity="sentence")

    def test_missing_vllm_raises_import_error(self):
        sys.modules.pop("vllm", None)
        sys.modules.pop("rmatch.segmenters", None)
        with pytest.raises(ImportError, match="vllm"):
            from rmatch.segmenters import SegmenterCuda

            SegmenterCuda(model_name="x")

    def test_pad_token_id_set_from_eos_when_none(self):
        mock_vllm = _make_vllm_module()
        tok = mock_vllm.LLM.return_value.get_tokenizer.return_value
        tok.pad_token_id = None
        tok.eos_token_id = 99
        with patch.dict(
            "sys.modules", {"vllm": mock_vllm, "torch": _make_torch_module()}
        ):
            from rmatch.segmenters import SegmenterCuda

            seg = SegmenterCuda(model_name="x")
        assert seg.tokenizer.pad_token_id == 99

    def test_hf_token_written_to_env(self, monkeypatch):
        import os

        mock_vllm = _make_vllm_module()
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch.dict(
            "sys.modules", {"vllm": mock_vllm, "torch": _make_torch_module()}
        ):
            from rmatch.segmenters import SegmenterCuda

            SegmenterCuda(model_name="x", api_key="hf-test-token")
        assert os.environ.get("HF_TOKEN") == "hf-test-token"


# ── SegmenterCuda._build_prompt ───────────────────────────────────────────────


class TestBuildPrompt:
    def test_idea_prompt_contains_input_text(self, segmenter):
        text = "She walked into the forest."
        assert text in segmenter._build_prompt(text)

    def test_event_prompt_contains_input_text(self, event_segmenter):
        text = "She walked into the forest."
        assert text in event_segmenter._build_prompt(text)

    def test_both_granularities_request_json_array(self, segmenter, event_segmenter):
        for seg in (segmenter, event_segmenter):
            prompt = seg._build_prompt("Some text.")
            assert "JSON array" in prompt or '["' in prompt

    def test_attempt_0_shorter_than_attempt_1(self, segmenter):
        p0 = segmenter._build_prompt("text", attempt=0)
        p1 = segmenter._build_prompt("text", attempt=1)
        assert len(p0) < len(p1)

    def test_attempt_1_mentions_verbatim(self, segmenter):
        assert "verbatim" in segmenter._build_prompt("text", attempt=1).lower()

    def test_attempt_2_mentions_careful(self, segmenter):
        assert "careful" in segmenter._build_prompt("text", attempt=2).lower()

    def test_high_attempt_number_appears_in_prompt(self, segmenter):
        assert "7" in segmenter._build_prompt("text", attempt=7)


# ── SegmenterCuda._validate_and_repair_segments ───────────────────────────────


class TestValidateAndRepairSegments:
    """Covers all significant branches in _validate_and_repair_segments."""

    def _v(self, segmenter, text, segments):
        return segmenter._validate_and_repair_segments(text, segments)

    # ── fully valid paths ─────────────────────────────────────────────────────

    def test_exact_match_is_valid(self, segmenter):
        text = "Hello world. Goodbye world."
        segs = ["Hello world.", " Goodbye world."]
        valid, failures = self._v(segmenter, text, segs)
        assert valid
        assert failures == {}

    def test_exact_match_without_spaces_is_valid(self, segmenter):
        text = "Hello world. Goodbye world."
        segs = ["Hello world.", "Goodbye world."]
        valid, failures = self._v(segmenter, text, segs)
        assert valid
        assert failures == {}

    def test_single_segment_covering_full_text(self, segmenter):
        text = "One complete sentence."
        valid, failures = self._v(segmenter, text, ["One complete sentence."])
        assert valid
        assert failures == {}

    def test_many_segments_all_valid(self, segmenter):
        parts = ["Seg one.", " Seg two.", " Seg three.", " Seg four.", " Seg five."]
        valid, failures = self._v(segmenter, "".join(parts), parts[:])
        assert valid
        assert failures == {}

    def test_duplicate_adjacent_word_handled_correctly(self, segmenter):
        # Prompt instructs the model to preserve duplicated words; validator must
        # not trip over two identical substrings appearing consecutively.
        text = "the the cat sat."
        valid, failures = self._v(segmenter, text, ["the the cat", "sat."])
        assert valid

    # ── error type labels ─────────────────────────────────────────────────────

    def test_bad_segment_gets_invalid_error_type(self, segmenter):
        _, failures = self._v(segmenter, "Alpha. Beta.", ["WRONG.", " Beta."])
        assert failures[0][1] == "invalid"

    def test_single_error_gets_invalid_error_type(self, segmenter):
        # Segment 1 is bad; segment 2 allows recovery → segment 1 is "invalid"
        _, failures = self._v(
            segmenter, "Alpha. Beta. Gamma.", ["Alpha.", "WRONG.", " Gamma."]
        )
        assert failures[1][1] == "invalid"

    def test_unrecoverable_failure_marks_remaining_as_unreachable(self, segmenter):
        # All segments wrong → segment 0 invalid, 1 and 2 unreachable
        _, failures = self._v(
            segmenter,
            "Alpha. Beta. Gamma.",
            ["TOTALLY WRONG.", "ALSO WRONG.", "STILL WRONG."],
        )
        assert failures[0][1] == "invalid"
        assert failures[1][1] == "unreachable"
        assert failures[2][1] == "unreachable"

    def test_remainder_error_type_string(self, segmenter):
        text = "Part one. Part two."
        segs = ["Part one."]
        _, failures = self._v(segmenter, text, segs)
        assert failures[len(segs)][1] == "original text not fully covered"

    # ── recovery logic ────────────────────────────────────────────────────────

    def test_recovery_skips_one_bad_segment(self, segmenter):
        text = "Alpha. Beta. Gamma."
        segs = ["Alpha.", "TOTALLY WRONG.", "Gamma."]
        _, failures = self._v(segmenter, text, segs)
        assert 1 in failures
        assert 0 not in failures
        assert 2 not in failures

    def test_recovery_skips_multiple_consecutive_bad_segments(self, segmenter):
        # Segments 1 and 2 bad; segment 3 allows recovery → 1=invalid, 2=skipped
        text = "A. B. C. D."
        segs = ["A.", "WRONG1.", "WRONG2.", "D."]
        _, failures = self._v(segmenter, text, segs)
        assert failures[1][1] == "invalid"
        assert failures[2][1] == "skipped"
        assert 0 not in failures
        assert 3 not in failures

    def test_cursor_advances_correctly_after_recovery(self, segmenter):
        # Segment after the recovered one should still be found
        text = "Alpha. Beta. Gamma. Delta."
        segs = ["Alpha.", "WRONG.", "Gamma.", "Delta."]
        _, failures = self._v(segmenter, text, segs)
        assert 2 not in failures
        assert 3 not in failures

    def test_multiple_independent_recoveries(self, segmenter):
        text = "A. B. C. D. E."
        segs = ["A.", "BAD.", " C.", "BAD2.", " E."]
        _, failures = self._v(segmenter, text, segs)
        assert 1 in failures
        assert 3 in failures
        assert 0 not in failures
        assert 2 not in failures
        assert 4 not in failures

    def test_bad_last_segment_is_invalid_not_unreachable(self, segmenter):
        # Nothing after the last segment to recover to → invalid, not unreachable
        text = "Alpha. Beta."
        _, failures = self._v(segmenter, text, ["Alpha.", "WRONG LAST."])
        assert 1 in failures
        assert all(failures[k][1] != "unreachable" for k in failures)

    # ── case-insensitive repair ───────────────────────────────────────────────

    def test_case_mismatch_repaired_in_place(self, segmenter):
        text = "The Quick Brown Fox."
        segs = ["the quick brown fox."]
        valid, _ = self._v(segmenter, text, segs)
        assert valid
        assert segs[0] == "The Quick Brown Fox."

    def test_case_repair_advances_cursor_so_next_segment_found(self, segmenter):
        text = "The Cat. Sat Down."
        segs = ["the cat.", " Sat Down."]
        valid, failures = self._v(segmenter, text, segs)
        assert valid
        assert failures == {}

    # ── SLACK boundary ────────────────────────────────────────────────────────

    def test_segment_found_exactly_at_cursor(self, segmenter):
        valid, _ = self._v(segmenter, "AB", ["A", "B"])
        assert valid

    def test_short_remainder_within_slack_not_an_error(self, segmenter):
        # SLACK=2: a 1-char remainder should not produce a remainder failure
        text = "Hello world.X"
        segs = ["Hello world."]
        _, failures = self._v(segmenter, text, segs)
        assert len(segs) not in failures

    def test_long_remainder_beyond_slack_is_an_error(self, segmenter):
        text = "Hello world. This is left over."
        segs = ["Hello world."]
        _, failures = self._v(segmenter, text, segs)
        assert len(segs) in failures

    # ── whitespace normalisation ──────────────────────────────────────────────

    def test_multiple_spaces_in_source_normalised(self, segmenter):
        text = "Hello   world.   Goodbye   world."
        valid, _ = self._v(segmenter, text, ["Hello world.", " Goodbye world."])
        assert valid

    def test_leading_trailing_whitespace_stripped(self, segmenter):
        valid, _ = self._v(segmenter, "  Hello world.  ", ["Hello world."])
        assert valid

    # ── empty segments list ───────────────────────────────────────────────────

    def test_empty_segments_with_nonempty_text_fails(self, segmenter):
        valid, failures = self._v(segmenter, "Some text.", [])
        assert not valid
        assert failures


# ── SegmenterCuda.segment / segment_batch ─────────────────────────────────────


class TestSegmentBatch:
    def _set_response(self, segmenter, texts: list[str]):
        segmenter.llm.generate.return_value = [_make_vllm_output(t) for t in texts]

    def test_bad_label_count_falls_back_to_indices(self, segmenter):
        segmenter.llm.generate.return_value = [
            _make_vllm_output(json.dumps(["Hello."]))
        ]

        result = segmenter.segment_batch(
            ["Hello."],
            labels=["A", "B"],
        )

        assert result[0] is not None

    # ── output shape ─────────────────────────────────────────────────────────

    def test_segment_returns_dataframe_with_correct_columns(self, segmenter):
        self._set_response(segmenter, [json.dumps(["Hello world."])])
        df = segmenter.segment("Hello world.")
        assert isinstance(df, pd.DataFrame)
        assert list(df["segment"]) == ["Hello world."]
        assert list(df["idx"]) == [1]
        assert not df["failed"].any()

    def test_batch_returns_one_dataframe_per_input(self, segmenter):
        texts = ["Hello world.", "Goodbye world."]
        segmenter.llm.generate.return_value = [
            _make_vllm_output(json.dumps(["Hello world."])),
            _make_vllm_output(json.dumps(["Goodbye world."])),
        ]
        results = segmenter.segment_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, pd.DataFrame) for r in results)

    def test_empty_batch_returns_empty_list(self, segmenter):
        assert segmenter.segment_batch([]) == []

    # ── retry orchestration ───────────────────────────────────────────────────

    def test_bad_json_triggers_retry(self, segmenter):
        segmenter.llm.generate.side_effect = [
            [_make_vllm_output("not json at all")],
            [_make_vllm_output(json.dumps(["Hello world."]))],
        ]
        segmenter.segment("Hello world.")
        assert segmenter.llm.generate.call_count == 2

    def test_validation_failure_triggers_retry(self, segmenter):
        segmenter.llm.generate.side_effect = [
            [_make_vllm_output(json.dumps(["WRONG.", "WRONG TOO."]))],
            [_make_vllm_output(json.dumps(["Alpha.", " Beta."]))],
        ]
        df = segmenter.segment("Alpha. Beta.")
        assert segmenter.llm.generate.call_count == 2
        assert not df["failed"].any()

    def test_max_retries_respected(self, segmenter):
        segmenter.max_retries = 4
        segmenter.llm.generate.return_value = [_make_vllm_output("bad json")]
        segmenter.segment("Hello.")
        assert segmenter.llm.generate.call_count == 4

    def test_exhausted_retries_returns_dataframe_not_none(self, segmenter):
        segmenter.llm.generate.return_value = [
            _make_vllm_output(json.dumps(["WRONG."]))
        ]
        df = segmenter.segment("Hello.")
        assert isinstance(df, pd.DataFrame)

    def test_batch_only_retries_failing_texts(self, segmenter):
        segmenter.llm.generate.side_effect = [
            [
                _make_vllm_output(json.dumps(["Good text."])),
                _make_vllm_output("bad json"),
            ],
            [_make_vllm_output(json.dumps(["Bad text."]))],
        ]
        results = segmenter.segment_batch(["Good text.", "Bad text."])
        assert segmenter.llm.generate.call_count == 2
        assert all(r is not None for r in results)

    # ── output content ────────────────────────────────────────────────────────

    def test_markdown_code_fence_stripped(self, segmenter):
        wrapped = "```json\n" + json.dumps(["Hello world."]) + "\n```"
        self._set_response(segmenter, [wrapped])
        df = segmenter.segment("Hello world.")
        assert list(df["segment"]) == ["Hello world."]

    def test_failed_rows_have_error_columns_populated(self, segmenter):
        segmenter.max_retries = 1
        self._set_response(segmenter, [json.dumps(["WRONG.", "WRONG."])])
        df = segmenter.segment("Alpha. Beta.")
        failed = df[df["failed"]]
        assert not failed.empty
        assert failed["error_type"].notna().any()
        assert failed["error_msg"].notna().any()
