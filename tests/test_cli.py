"""Tests for the rmatch CLI entry point (rmatch.matching:main).

Subprocess-based tests verify argument parsing without requiring the full
dependency stack.  Direct main() invocations patch the Matcher so no real API
calls are made.
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import RECALL_SEGMENTS

# ── Subprocess / argparse tests (no API calls) ────────────────────────────────


@pytest.mark.integration
class TestCLIArgparse:
    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "rmatch.matching", *args],
            capture_output=True,
            text=True,
        )

    def test_help_exits_zero(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "story_file" in result.stdout

    def test_missing_positional_args_exits_nonzero(self):
        result = self._run([])
        assert result.returncode != 0

    def test_missing_recall_file_exits_nonzero(self):
        result = self._run(["/tmp/story.txt"])
        assert result.returncode != 0

    def test_invalid_matcher_choice_exits_nonzero(self, story_txt, recall_txt):
        result = self._run([str(story_txt), str(recall_txt), "-M", "invalid_matcher"])
        assert result.returncode != 0
        combined = result.stderr + result.stdout
        assert "invalid choice" in combined.lower() or "error" in combined.lower()


# ── Direct main() invocation with mocked Matcher ─────────────────────────────


def _make_mock_matcher(n_recall: int = 2):
    m = MagicMock()
    m.match.return_value = [(i, []) for i in range(n_recall)]
    m.matcher_name = "anthropic"
    m.model_name = None
    return m


class TestCLIMain:
    def test_main_nonexistent_story_exits_with_error(self, recall_txt):
        with patch(
            "sys.argv",
            ["rmatch", "/nonexistent/story.txt", str(recall_txt), "-M", "anthropic"],
        ):
            from rmatch.matching import main

            with pytest.raises((FileNotFoundError, SystemExit)):
                main()

    def test_main_openai_matcher(self, story_txt, recall_txt):
        mock_matcher = _make_mock_matcher()
        mock_matcher.matcher_name = "openai"
        with patch("rmatch.matching.get_matcher", return_value=mock_matcher):
            with patch(
                "sys.argv",
                [
                    "rmatch",
                    str(story_txt),
                    str(recall_txt),
                    "-M",
                    "openai",
                    "-f",
                ],
            ):
                from rmatch.matching import main

                main()
        mock_matcher.match.assert_called_once()

    def test_main_dry_run_anthropic(self, story_txt, recall_txt):
        """dry_run skips the real API call and uses count_tokens instead."""
        mock_client = MagicMock()
        mock_count = MagicMock()
        mock_count.input_tokens = 42
        mock_client.messages.count_tokens.return_value = mock_count

        with patch("anthropic.Anthropic", return_value=mock_client):
            with patch(
                "sys.argv",
                [
                    "rmatch",
                    str(story_txt),
                    str(recall_txt),
                    "-M",
                    "anthropic",
                    "--dry-run",
                    "-f",
                ],
            ):
                from rmatch.matching import main

                main()  # Should not raise; no real API call made

    @pytest.mark.parametrize(
        "extra_argv,expected_key,expected_value",
        [
            (["-m", "foo"], "model_name", "foo"),
            (["--window-size", "3"], "window_size", 3),
            (["-q", "4bit"], "quantization", "4bit"),
            (["-bs", "8"], "batch_size", 8),
            (["--max-new-tokens", "128"], "max_new_tokens", 128),
            (["--max-model-len", "4096"], "max_model_len", 4096),
            (["--verbose-errors"], "verbose_errors", True),
            (["--device", "cpu"], "device", "cpu"),
            (["--prompt", "secondary"], "prompt", "secondary"),
        ],
    )
    def test_main_forwards_matcher_kwargs(
        self, story_txt, recall_txt, extra_argv, expected_key, expected_value
    ):
        mock_matcher = _make_mock_matcher()
        captured_kwargs = {}

        def capture_matcher(name, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_matcher

        with patch("rmatch.matching.get_matcher", side_effect=capture_matcher):
            with patch(
                "sys.argv",
                [
                    "rmatch",
                    str(story_txt),
                    str(recall_txt),
                    "-M",
                    "anthropic",
                    *extra_argv,
                    "-f",
                ],
            ):
                from rmatch.matching import main

                main()

        assert captured_kwargs.get(expected_key) == expected_value
