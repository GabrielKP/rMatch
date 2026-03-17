import os
import unittest
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

    @patch("recall_matrix.rate_binary.ENV")
    def test_no_API_key(self, mock_env):
        mock_env.get.return_value = None
        print(ENV.get("HF_TOKEN"))
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


if __name__ == "__main__":
    unittest.main()
