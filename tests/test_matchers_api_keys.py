"""Tests that all three LLM matchers validate their API keys correctly.

Key pattern: `load_dotenv()` runs at rmatch package import time, so any key
present in a local .env file would normally bypass these checks.  We patch
the `os` module *inside each matcher module* (not globally) so that
os.environ.get(...) returns exactly what the test dictates.
"""

from unittest.mock import MagicMock, patch

import pytest

# ── MatcherAnthropic ──────────────────────────────────────────────────────────


class TestAnthropicApiKey:
    def test_missing_key_raises_value_error(self):
        with patch("rmatch.matchers.matcher_anthropic.os") as mock_os:
            mock_os.environ.get.return_value = None
            from rmatch.matchers.matcher_anthropic import MatcherAnthropic

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                MatcherAnthropic(api_key=None)

    def test_explicit_key_bypasses_env_lookup(self):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            from rmatch.matchers.matcher_anthropic import MatcherAnthropic

            # Should not raise even if env var is absent
            m = MatcherAnthropic(api_key="sk-explicit-key")
            mock_cls.assert_called_once_with(api_key="sk-explicit-key")
            assert m is not None

    def test_env_var_key_used_when_no_explicit_key(self):
        with patch("rmatch.matchers.matcher_anthropic.os") as mock_os:
            mock_os.environ.get.return_value = "sk-env-key-123"
            with patch("anthropic.Anthropic") as mock_cls:
                mock_cls.return_value = MagicMock()
                from rmatch.matchers.matcher_anthropic import MatcherAnthropic

                m = MatcherAnthropic(api_key=None)
                mock_cls.assert_called_once_with(api_key="sk-env-key-123")
                assert m is not None

    def test_missing_key_error_message_is_helpful(self):
        with patch("rmatch.matchers.matcher_anthropic.os") as mock_os:
            mock_os.environ.get.return_value = None
            from rmatch.matchers.matcher_anthropic import MatcherAnthropic

            with pytest.raises(ValueError) as exc_info:
                MatcherAnthropic(api_key=None)
            # Message should hint at where to set the key
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)
            assert ".env" in str(exc_info.value) or "environment" in str(exc_info.value)


# ── MatcherOpenAI ─────────────────────────────────────────────────────────────


class TestOpenAIApiKey:
    def test_missing_key_raises_value_error(self):
        with patch("rmatch.matchers.matcher_openai.os") as mock_os:
            mock_os.environ.get.return_value = None
            from rmatch.matchers.matcher_openai import MatcherOpenAI

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                MatcherOpenAI(api_key=None)

    def test_explicit_key_bypasses_env_lookup(self):
        with patch("rmatch.matchers.matcher_openai.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            from rmatch.matchers.matcher_openai import MatcherOpenAI

            m = MatcherOpenAI(api_key="sk-openai-explicit")
            mock_cls.assert_called_once_with(api_key="sk-openai-explicit")
            assert m is not None

    def test_env_var_key_used_when_no_explicit_key(self):
        with patch("rmatch.matchers.matcher_openai.os") as mock_os:
            mock_os.environ.get.return_value = "sk-openai-env"
            with patch("rmatch.matchers.matcher_openai.OpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                from rmatch.matchers.matcher_openai import MatcherOpenAI

                m = MatcherOpenAI(api_key=None)
                mock_cls.assert_called_once_with(api_key="sk-openai-env")
                assert m is not None

    def test_missing_key_error_message_is_helpful(self):
        with patch("rmatch.matchers.matcher_openai.os") as mock_os:
            mock_os.environ.get.return_value = None
            from rmatch.matchers.matcher_openai import MatcherOpenAI

            with pytest.raises(ValueError) as exc_info:
                MatcherOpenAI(api_key=None)
            assert "OPENAI_API_KEY" in str(exc_info.value)


# ── MatcherHuggingFace ────────────────────────────────────────────────────────


class TestHuggingFaceApiKey:
    def _make_pipe_mock(self):
        pipe = MagicMock()
        pipe.tokenizer.pad_token_id = None
        pipe.tokenizer.eos_token_id = 2
        pipe.tokenizer.padding_side = "right"
        return pipe

    def test_missing_key_raises_value_error(self):
        with patch("rmatch.matchers.matcher_huggingface.os") as mock_os:
            mock_os.environ.get.return_value = None
            with patch("rmatch.matchers.matcher_huggingface.pipeline"):
                from rmatch.matchers.matcher_huggingface import MatcherHuggingFace

                with pytest.raises(ValueError, match="HF_TOKEN"):
                    MatcherHuggingFace(api_key=None)

    def test_explicit_key_bypasses_env_lookup(self):
        pipe = self._make_pipe_mock()
        with patch("rmatch.matchers.matcher_huggingface.pipeline", return_value=pipe):
            from rmatch.matchers.matcher_huggingface import MatcherHuggingFace

            m = MatcherHuggingFace(api_key="hf-explicit-token")
            assert m is not None

    def test_env_var_key_used_when_no_explicit_key(self):
        pipe = self._make_pipe_mock()
        with patch("rmatch.matchers.matcher_huggingface.os") as mock_os:
            mock_os.environ.get.return_value = "hf-env-token"
            with patch(
                "rmatch.matchers.matcher_huggingface.pipeline", return_value=pipe
            ) as mock_pipeline:
                from rmatch.matchers.matcher_huggingface import MatcherHuggingFace

                m = MatcherHuggingFace(api_key=None)
                assert m is not None
                # pipeline should have been called with the env token
                call_kwargs = mock_pipeline.call_args[1]
                assert call_kwargs.get("token") == "hf-env-token"

    def test_missing_key_error_message_is_helpful(self):
        with patch("rmatch.matchers.matcher_huggingface.os") as mock_os:
            mock_os.environ.get.return_value = None
            with patch("rmatch.matchers.matcher_huggingface.pipeline"):
                from rmatch.matchers.matcher_huggingface import MatcherHuggingFace

                with pytest.raises(ValueError) as exc_info:
                    MatcherHuggingFace(api_key=None)
                assert "HF_TOKEN" in str(exc_info.value)
