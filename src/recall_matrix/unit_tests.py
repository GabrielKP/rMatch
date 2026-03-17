import unittest

from recall_matrix.rate_binary import rate_binary


class TestRateBinary(unittest.TestCase):
    def test_wrong_input_directory(self):
        with self.assertRaises(FileNotFoundError):
            rate_binary(
                rater_name="huggingface",
                story_name="nonexistent_story",
                story_segmentation_method="sentences",
                recall_segmentation_method="sentences",
                output_scores=True,
                sub_ids=["sub1", "sub2"],
                model_name="test_model",
                device="cpu",
                reranker_threshold=0.5,
                top_k=3,
            )

    def test_wrong_model_name(self):
        with self.assertRaises(ValueError):
            rate_binary(
                rater_name="huggingface",
                story_name="pieman",
                story_segmentation_method="sentences",
                recall_segmentation_method="sentences",
                output_scores=True,
                sub_ids=["sub1", "sub2"],
                model_name="nonexistent_model",
                device="cpu",
                reranker_threshold=0.5,
                top_k=3,
            )


if __name__ == "__main__":
    unittest.main()
