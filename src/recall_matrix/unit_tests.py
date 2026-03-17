import os
import unittest
from unittest import mock
from unittest.mock import patch

from recall_matrix import ENV
from recall_matrix.rate_binary import rate_binary


class TestRateBinary(unittest.TestCase):
    def test_wrong_input_directory(self):
        with self.assertRaises(FileNotFoundError):
            rate_binary(
                rater_name="reranker",
                story_name="nonexistent_story",
                story_segmentation_method="sentences",
                recall_segmentation_method="sentences",
                output_scores=True,
                sub_ids=["sub-001", "sub-002"],
                model_name=None,
                device=None,
                reranker_threshold=0.5,
                top_k=3,
            )

    def test_wrong_rater_name(self):
        with self.assertRaises(ValueError):
            rate_binary(
                rater_name="nonexistent_rater",
                story_name="pieman",
                story_segmentation_method="sentences",
                recall_segmentation_method="sentences",
                output_scores=True,
                sub_ids=["sub-001", "sub-002"],
                model_name=None,
                device=None,
                reranker_threshold=0.5,
                top_k=3,
            )

    def test_no_API_key(self):
        # Ensure missing API keys trigger a failure.
        # Patch ENV so that tokens are present but empty/None (no real token).
        with patch.dict(
            "recall_matrix.ENV",
            {"HF_TOKEN": None, "OPENAI_API_KEY": None},
            clear=True,
        ):
            # Reranker should fail if HF_TOKEN is empty/None.
            with patch("recall_matrix.raters.rater_reranker.CrossEncoder") as mock_ce:
                mock_ce.side_effect = ValueError("Missing HF_TOKEN")
                with self.assertRaises(ValueError):
                    rate_binary(
                        rater_name="reranker",
                        story_name="pieman",
                        story_segmentation_method="sentences",
                        recall_segmentation_method="sentences",
                        output_scores=True,
                        sub_ids=["sub-001", "sub-002"],
                        model_name=None,
                        device=None,
                        reranker_threshold=0.5,
                        top_k=3,
                    )

            # OpenAI should fail if OPENAI_API_KEY is empty/None.
            with patch("recall_matrix.raters.rater_openai.OpenAI") as mock_openai:
                mock_openai.side_effect = ValueError("Missing OPENAI_API_KEY")
                with self.assertRaises(ValueError):
                    rate_binary(
                        rater_name="openai",
                        story_name="pieman",
                        story_segmentation_method="sentences",
                        recall_segmentation_method="sentences",
                        output_scores=True,
                        sub_ids=["sub-001", "sub-002"],
                        model_name=None,
                        device=None,
                        reranker_threshold=0.5,
                        top_k=3,
                    )


if __name__ == "__main__":
    unittest.main()
