"""Tests for rmatch.evaluate: accuracy() and the evaluate() function.

evaluate() requires a complex benchmark directory structure and real matcher
calls.  Rather than building all that infrastructure, the heavy lifting is
mocked: load_benchmark_full_eval returns synthetic data, and Matcher returns a
pre-built matchlist.  This lets us focus on the logic that evaluate() itself
is responsible for (checkpoint writing, dry-run early-return, metrics).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rmatch.evaluate import accuracy
from tests.conftest import RECALL_SEGMENTS, STORY_SEGMENTS

# ── accuracy() ───────────────────────────────────────────────────────────────


class TestAccuracy:
    def test_all_correct(self):
        a = np.array([1, 0, 1, 0])
        assert accuracy(a, a) == pytest.approx(1.0)

    def test_all_wrong(self):
        assert accuracy(np.array([1, 1]), np.array([0, 0])) == pytest.approx(0.0)

    def test_partial(self):
        # 3 out of 4 correct
        a = np.array([1, 0, 1, 0])
        b = np.array([1, 0, 1, 1])
        assert accuracy(a, b) == pytest.approx(0.75)

    def test_empty_arrays_returns_zero(self):
        assert accuracy(np.array([]), np.array([])) == pytest.approx(0.0)

    def test_single_element_match(self):
        assert accuracy(np.array([1]), np.array([1])) == pytest.approx(1.0)

    def test_single_element_mismatch(self):
        assert accuracy(np.array([1]), np.array([0])) == pytest.approx(0.0)

    def test_all_zeros_equal(self):
        a = np.zeros(5, dtype=int)
        assert accuracy(a, a.copy()) == pytest.approx(1.0)


# ── evaluate() dry-run path ────────────────────────────────────────────────────


def _build_human_data(n_story: int, n_recall: int):
    """Build a human match matrix and matchlist that are NOT all-zero."""
    matrix = np.zeros((n_story, n_recall), dtype=int)
    matrix[0, 0] = 1  # at least one match
    matchlist = [(i, [0] if i == 0 else []) for i in range(n_recall)]
    return matrix, matchlist


@pytest.fixture
def simple_benchmark_data():
    """Minimal story/recall data for evaluate() tests."""
    n_story = len(STORY_SEGMENTS)
    n_recall = len(RECALL_SEGMENTS)
    human_matrix, human_matchlist = _build_human_data(n_story, n_recall)

    story_recall_segments = [
        ("story1", "sub01", STORY_SEGMENTS, RECALL_SEGMENTS),
    ]
    human_ratings_dict = {
        "story1": {
            "sub01": (human_matrix, human_matchlist),
        }
    }
    return story_recall_segments, human_ratings_dict


@pytest.fixture
def mock_eval_matcher():
    """A minimal mock Matcher for evaluate() tests."""
    m = MagicMock()
    m.match.return_value = [(i, []) for i in range(len(RECALL_SEGMENTS))]
    # prompt_response_log must be indexable for save_raw_response
    m.prompt_response_log = {
        "story1_sub01_0": [
            [("prompt_text", "response_text", None)]
            for _ in range(len(RECALL_SEGMENTS))
        ]
    }
    m.get_usage.return_value = None
    m.model_name = None
    return m


class TestEvaluateDryRun:
    def test_dry_run_does_not_write_results_json(
        self, tmp_path, simple_benchmark_data, mock_eval_matcher, monkeypatch
    ):
        """evaluate(..., dry_run=True) should return before writing results.json."""
        monkeypatch.chdir(tmp_path)
        story_recall_segments, human_ratings_dict = simple_benchmark_data

        with patch(
            "rmatch.evaluate.load_benchmark_full_eval",
            return_value=(story_recall_segments, human_ratings_dict),
        ):
            with patch("rmatch.evaluate.Matcher", return_value=mock_eval_matcher):
                from rmatch.evaluate import evaluate

                evaluate(
                    testset="alice",
                    benchmark_root=tmp_path,
                    matcher_name="anthropic",
                    dry_run=True,
                )

        results_files = list(tmp_path.rglob("results.json"))
        assert len(results_files) == 0

    def test_dry_run_calls_matcher_match(
        self, tmp_path, simple_benchmark_data, mock_eval_matcher, monkeypatch
    ):
        """Even in dry_run mode, the matcher is called so we can estimate cost."""
        monkeypatch.chdir(tmp_path)
        story_recall_segments, human_ratings_dict = simple_benchmark_data

        with patch(
            "rmatch.evaluate.load_benchmark_full_eval",
            return_value=(story_recall_segments, human_ratings_dict),
        ):
            with patch("rmatch.evaluate.Matcher", return_value=mock_eval_matcher):
                from rmatch.evaluate import evaluate

                evaluate(
                    testset="alice",
                    benchmark_root=tmp_path,
                    matcher_name="anthropic",
                    dry_run=True,
                )

        mock_eval_matcher.match.assert_called_once()

    def test_non_dry_run_writes_results_json(
        self, tmp_path, simple_benchmark_data, mock_eval_matcher, monkeypatch
    ):
        """Without dry_run, results.json must exist after evaluate() completes."""
        monkeypatch.chdir(tmp_path)
        story_recall_segments, human_ratings_dict = simple_benchmark_data

        with patch(
            "rmatch.evaluate.load_benchmark_full_eval",
            return_value=(story_recall_segments, human_ratings_dict),
        ):
            with patch("rmatch.evaluate.Matcher", return_value=mock_eval_matcher):
                from rmatch.evaluate import evaluate

                evaluate(
                    testset="alice",
                    benchmark_root=tmp_path,
                    matcher_name="anthropic",
                    dry_run=False,
                )

        results_files = list(tmp_path.rglob("results.json"))
        assert len(results_files) == 1

    def test_results_json_contains_expected_keys(
        self, tmp_path, simple_benchmark_data, mock_eval_matcher, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        story_recall_segments, human_ratings_dict = simple_benchmark_data

        with patch(
            "rmatch.evaluate.load_benchmark_full_eval",
            return_value=(story_recall_segments, human_ratings_dict),
        ):
            with patch("rmatch.evaluate.Matcher", return_value=mock_eval_matcher):
                from rmatch.evaluate import evaluate

                evaluate(
                    testset="alice",
                    benchmark_root=tmp_path,
                    matcher_name="anthropic",
                    dry_run=False,
                )

        results_file = next(tmp_path.rglob("results.json"))
        data = json.loads(results_file.read_text())
        for key in (
            "testset",
            "matcher_name",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ):
            assert key in data, f"Missing key: {key}"

    def test_invalid_testset_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("rmatch.evaluate.Matcher", return_value=MagicMock()):
            from rmatch.evaluate import evaluate

            with pytest.raises(ValueError, match="Invalid testset"):
                evaluate(
                    testset="nonexistent_testset",
                    benchmark_root=tmp_path,
                    matcher_name="anthropic",
                )

    def test_all_zero_human_matrix_skips_recall(
        self, tmp_path, mock_eval_matcher, monkeypatch
    ):
        """A recall whose human matrix is all-zero is skipped
        (n_skipped incremented)."""
        monkeypatch.chdir(tmp_path)
        n_story, n_recall = len(STORY_SEGMENTS), len(RECALL_SEGMENTS)
        # All-zero human matrix
        zero_matrix = np.zeros((n_story, n_recall), dtype=int)
        matchlist = [(i, []) for i in range(n_recall)]

        story_recall_segments = [("story1", "sub01", STORY_SEGMENTS, RECALL_SEGMENTS)]
        human_ratings_dict = {"story1": {"sub01": (zero_matrix, matchlist)}}

        with patch(
            "rmatch.evaluate.load_benchmark_full_eval",
            return_value=(story_recall_segments, human_ratings_dict),
        ):
            with patch("rmatch.evaluate.Matcher", return_value=mock_eval_matcher):
                from rmatch.evaluate import evaluate

                # Should raise ValueError because no non-skipped recalls remain
                with pytest.raises(ValueError):
                    evaluate(
                        testset="alice",
                        benchmark_root=tmp_path,
                        matcher_name="anthropic",
                        dry_run=False,
                    )

        # Matcher should NOT have been called for the skipped recall
        mock_eval_matcher.match.assert_not_called()
