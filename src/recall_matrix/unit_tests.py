import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np

from recall_matrix import ENV
from recall_matrix.evaluate_binary import (
    accuracy,
    eval_param_str,
    evaluate,
    get_average_pairwise_f1,
    get_krippendorff_alpha,
    get_model_ratings,
)
from recall_matrix.rate_binary import rate_binary
from recall_matrix.raters import initialize_rater
from recall_matrix.raters.rater import Rater
from recall_matrix.raters.rater_openai import RaterOpenAI
from recall_matrix.raters.rater_reranker import RaterReranker


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
        with patch.dict(
            "recall_matrix.ENV",
            {"HF_TOKEN": None, "OPENAI_API_KEY": None},
            clear=True,
        ):
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


class DummyRaterForTests(Rater):
    def __init__(self):
        super().__init__()
        self.rater_name = "dummy"
        self.calls = []

    def compute_ratings_single_sub(
        self,
        story_segments: list[str],
        recall_segments: list[str],
        output_scores: bool = False,
    ) -> list[tuple[int, list[int] | list[tuple[int, float]]]]:
        self.calls.append((story_segments, recall_segments))
        return [(idx, []) for idx in range(len(recall_segments))]


class TestRater(unittest.TestCase):
    @patch("recall_matrix.raters.rater.load_story_recall_segments")
    def test_rater_accesses_file_hierarchy(self, mock_load):
        story_recall_segments = [
            ("sub-001", ["TRANSCRIPT A1", "TRANSCRIPT A2"], ["RECALL A1"]),
            ("sub-002", ["TRANSCRIPT B1"], ["RECALL B1", "RECALL B2"]),
        ]
        mock_load.return_value = (story_recall_segments, "sentences", "sentences")

        dummy = DummyRaterForTests()
        output = dummy.rate(
            story_name="dummy_story",
            story_segmentation_method="sentences",
            recall_segmentation_method="sentences",
            sub_ids=None,
            output_scores=False,
        )

        self.assertEqual(len(dummy.calls), 2)
        self.assertEqual(dummy.calls[0][0], ["TRANSCRIPT A1", "TRANSCRIPT A2"])
        self.assertEqual(dummy.calls[1][0], ["TRANSCRIPT B1"])
        self.assertEqual(dummy.calls[0][1], ["RECALL A1"])
        self.assertEqual(dummy.calls[1][1], ["RECALL B1", "RECALL B2"])

        self.assertIn("sub-001", output["ratings"])
        self.assertIn("sub-002", output["ratings"])

    def test_output_json_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                rater = Rater()
                rater.rater_name = "dummy"
                output_dict = {
                    "rater_name": "dummy",
                    "story_name": "test_story",
                    "story_segmentation_method": "sentences",
                    "recall_segmentation_method": "sentences",
                    "n_story_segments": 1,
                    "output_scores": True,
                    "ratings": {"sub-001": [(0, [(0, 0.5)])]},
                }

                output_path = rater.save_to_json(output_dict)

                expected_filename = (
                    "dummy-ssm_sentences-rsm_sentences-ouput_scores.json"
                )
                self.assertEqual(output_path.name, expected_filename)
                self.assertEqual(
                    tuple(output_path.parts[-5:]),
                    (
                        "data",
                        "stories-and-recalls",
                        "test_story",
                        "ratings",
                        expected_filename,
                    ),
                )

                full_path = os.path.join(tmpdir, str(output_path))
                self.assertTrue(os.path.exists(full_path))

                with open(full_path, "r") as f:
                    loaded = json.load(f)
                self.assertEqual(loaded["rater_name"], "dummy")
                self.assertEqual(loaded["story_name"], "test_story")
            finally:
                os.chdir(cwd)

    def test_output_json_format(self):
        """Ensure the saved JSON has the expected schema and types."""

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                rater = Rater()
                rater.rater_name = "dummy"
                output_dict = {
                    "rater_name": "dummy",
                    "story_name": "format_story",
                    "story_segmentation_method": "sentences",
                    "recall_segmentation_method": "sentences",
                    "n_story_segments": 1,
                    "output_scores": True,
                    "ratings": {"sub-001": [(0, [(0, 0.5)])]},
                }

                output_path = rater.save_to_json(output_dict)
                full_path = os.path.join(tmpdir, str(output_path))

                with open(full_path, "r") as f:
                    loaded = json.load(f)

                # Validate top-level keys and value types.
                expected_keys = {
                    "rater_name",
                    "story_name",
                    "story_segmentation_method",
                    "recall_segmentation_method",
                    "n_story_segments",
                    "output_scores",
                    "ratings",
                }
                self.assertEqual(set(loaded.keys()), expected_keys)
                self.assertIsInstance(loaded["rater_name"], str)
                self.assertIsInstance(loaded["story_name"], str)
                self.assertIsInstance(loaded["story_segmentation_method"], str)
                self.assertIsInstance(loaded["recall_segmentation_method"], str)
                self.assertIsInstance(loaded["n_story_segments"], int)
                self.assertIsInstance(loaded["output_scores"], bool)

                ratings = loaded["ratings"]
                self.assertIsInstance(ratings, dict)

                for sub_id, subject_ratings in ratings.items():
                    self.assertIsInstance(sub_id, str)
                    self.assertIsInstance(subject_ratings, list)
                    for recall_entry in subject_ratings:
                        self.assertIsInstance(recall_entry, list)
                        self.assertEqual(len(recall_entry), 2)

                        recall_id, story_scores = recall_entry
                        self.assertIsInstance(recall_id, int)
                        self.assertIsInstance(story_scores, list)

                        for item in story_scores:
                            self.assertIsInstance(item, list)
                            self.assertEqual(len(item), 2)
                            story_id, score = item
                            self.assertIsInstance(story_id, int)
                            self.assertIsInstance(score, float)
            finally:
                os.chdir(cwd)

    @patch("recall_matrix.rate_binary.plot_single_sub")
    @patch("recall_matrix.rate_binary.initialize_rater")
    def test_rate_binary_finds_correct_rater(self, mock_init, mock_plot):
        dummy_rater = Mock()
        dummy_rater.rate.return_value = {
            "output_scores": False,
            "ratings": {"sub-001": [(0, [0])]},
        }
        dummy_rater.save_to_json.return_value = Path(
            "data/stories-and-recalls/foo/ratings/foo.json"
        )
        mock_init.return_value = dummy_rater

        rate_binary(
            rater_name="openai",
            story_name="foo",
            story_segmentation_method="sentences",
            recall_segmentation_method="sentences",
            output_scores=False,
            sub_ids=["sub-001"],
            model_name="m",
            device="cpu",
            reranker_threshold=0.5,
            top_k=3,
        )

        mock_init.assert_called_once_with(
            rater_name="openai",
            model_name="m",
            device="cpu",
        )
        dummy_rater.rate.assert_called_once()
        dummy_rater.save_to_json.assert_called_once()

        saved_dict = dummy_rater.save_to_json.call_args.args[0]
        self.assertIsInstance(saved_dict, dict)
        self.assertFalse(saved_dict["output_scores"])
        self.assertIn("sub-001", saved_dict["ratings"])
        mock_plot.assert_called_once()

    @patch("recall_matrix.rate_binary.plot_single_sub")
    @patch("recall_matrix.rate_binary.initialize_rater")
    def test_rate_binary_does_not_plot_when_output_scores_true(
        self, mock_init, mock_plot
    ):
        dummy_rater = Mock()
        dummy_rater.rate.return_value = {
            "output_scores": True,
            "ratings": {"sub-001": [(0, [(0, 0.5)])]},
        }
        dummy_rater.save_to_json.return_value = Path(
            "data/stories-and-recalls/foo/ratings/foo.json"
        )
        mock_init.return_value = dummy_rater

        rate_binary(
            rater_name="openai",
            story_name="foo",
            story_segmentation_method="sentences",
            recall_segmentation_method="sentences",
            output_scores=True,
            sub_ids=["sub-001"],
            model_name="m",
            device="cpu",
            reranker_threshold=0.5,
            top_k=3,
        )

        mock_plot.assert_not_called()
        dummy_rater.save_to_json.assert_called_once()

        saved_dict = dummy_rater.save_to_json.call_args.args[0]
        self.assertTrue(saved_dict["output_scores"])
        self.assertIn("sub-001", saved_dict["ratings"])

    @patch("recall_matrix.rate_binary.initialize_rater")
    def test_rate_binary_bad_output_causes_crash(self, mock_init):
        dummy_rater = Mock()
        dummy_rater.rate.return_value = {"output_scores": False}
        dummy_rater.save_to_json.return_value = Path(
            "data/stories-and-recalls/foo/ratings/foo.json"
        )
        mock_init.return_value = dummy_rater

        with self.assertRaises(KeyError):
            rate_binary(
                rater_name="openai",
                story_name="foo",
                story_segmentation_method="sentences",
                recall_segmentation_method="sentences",
                output_scores=False,
                sub_ids=["sub-001"],
                model_name="m",
                device="cpu",
                reranker_threshold=0.5,
                top_k=3,
            )


class TestOpenAIRater(unittest.TestCase):
    @patch("recall_matrix.raters.rater_openai.OpenAI")
    def test_openai_rater_builds_prompt_and_parses_output(self, mock_openai_cls):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "<1, 2>"
        mock_client.responses.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")
            ratings = rater.compute_ratings_single_sub(
                story_segments=["A", "B"],
                recall_segments=["recall"],
                output_scores=False,
            )

        # should produce a list with one recall entry, mapping to story indices 0 and 1
        self.assertEqual(ratings, [(0, [0, 1])])
        mock_client.responses.create.assert_called_once()
        kwargs = mock_client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-test")
        self.assertIn("This is the original text:", kwargs["input"])
        self.assertIn("recall", kwargs["input"])

    def test_openai_rater_refuses_output_scores(self):
        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")
            with self.assertRaises(NotImplementedError):
                rater.compute_ratings_single_sub(
                    story_segments=["A"],
                    recall_segments=["r"],
                    output_scores=True,
                )


class TestRerankerRater(unittest.TestCase):
    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_reranker_uses_threshold_and_top_k(self, mock_crossencoder):
        mock_model = Mock()
        # return more items than top_k to ensure top_k is passed through
        mock_model.rank.return_value = [
            {"score": 0.9, "corpus_id": 0},
            {"score": 0.7, "corpus_id": 1},
            {"score": 0.5, "corpus_id": 2},
        ]
        mock_crossencoder.return_value = mock_model

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            rater = RaterReranker(
                model_name="model-x", device="cpu", threshold=0.6, top_k=2
            )
            ratings = rater.compute_ratings_single_sub(
                story_segments=["s1", "s2", "s3"],
                recall_segments=["r1"],
                output_scores=False,
            )

        # Expect only scores above threshold and within top_k (first 2 entries)
        self.assertEqual(ratings, [(0, [0, 1])])
        mock_crossencoder.assert_called_once_with(
            "model-x", device="cpu", token="token"
        )
        mock_model.rank.assert_called_once_with("r1", ["s1", "s2", "s3"], top_k=2)

    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_reranker_outputs_scores_when_requested(self, mock_crossencoder):
        mock_model = Mock()
        mock_model.rank.return_value = [
            {"score": 0.9, "corpus_id": 0},
        ]
        mock_crossencoder.return_value = mock_model

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            rater = RaterReranker(
                model_name="model-x", device="cpu", threshold=0.0, top_k=1
            )
            ratings = rater.compute_ratings_single_sub(
                story_segments=["s1"],
                recall_segments=["r1"],
                output_scores=True,
            )

        self.assertEqual(ratings, [(0, [(0, 0.9)])])


class TestEvaluateBinary(unittest.TestCase):
    def test_eval_param_str_includes_expected_parts(self):
        fixed_dt = datetime.datetime(2020, 1, 2, 3, 4, 5)
        with patch(
            "recall_matrix.evaluate_binary.datetime.datetime", wraps=datetime.datetime
        ) as mock_dt:
            mock_dt.now.return_value = fixed_dt
            param_str = eval_param_str(
                repeat_reliability=True,
                testset="cyoa_alice10",
                rater_name="openai",
                model_name="m/foo",
                seed=123,
                random_mode="full_shuffle",
            )

        self.assertIn("20200102_030405", param_str)
        self.assertIn("-rr-", param_str)
        self.assertIn("-cyoa_alice10-", param_str)
        self.assertIn("-m_m_foo", param_str)
        self.assertIn("-random_mode_full_shuffle", param_str)

    def test_accuracy_empty_returns_zero(self):
        self.assertEqual(accuracy(np.array([]), np.array([])), 0.0)

    @patch("recall_matrix.evaluate_binary.initialize_rater")
    def test_get_model_ratings_random_permutations(self, mock_init):
        rm_comparison = np.array([[1, 0], [0, 1]])
        rng = np.random.default_rng(42)

        dummy_rater = DummyRaterForTests()
        mock_init.return_value = dummy_rater

        # full shuffle should preserve all values and shape, but permute them
        rm_full = get_model_ratings(
            random_mode="full_shuffle",
            rng=np.random.default_rng(42),
            rm_comparison=rm_comparison,
            rater=initialize_rater(
                rater_name="reranker", model_name="BAAI/bge-reranker-v2-m3"
            ),
            story_segments=[],
            recall_segments=[],
        )
        expected = rng.permutation(rm_comparison.flatten()).reshape(rm_comparison.shape)
        self.assertTrue(np.array_equal(rm_full, expected))

        # row shuffle should permute rows but keep each row values
        rng = np.random.default_rng(42)
        rm_row = get_model_ratings(
            random_mode="row_shuffle",
            rng=np.random.default_rng(42),
            rm_comparison=rm_comparison,
            rater=dummy_rater,
            story_segments=[],
            recall_segments=[],
        )
        self.assertEqual(rm_row.shape, rm_comparison.shape)
        self.assertTrue(
            np.array_equal(
                np.sort(rm_row, axis=None), np.sort(rm_comparison, axis=None)
            )
        )

    def test_get_model_ratings_rejects_non_rater(self):
        with self.assertRaises(TypeError):
            get_model_ratings(
                random_mode=None,
                rng=np.random.default_rng(42),
                rm_comparison=np.array([[1, 0], [0, 1]]),
                rater=0.1,  # type: ignore
                story_segments=["s1"],
                recall_segments=["r1"],
            )

    @patch("recall_matrix.evaluate_binary.load_story_recall_segments_default")
    @patch("recall_matrix.evaluate_binary.initialize_rater")
    @patch("recall_matrix.evaluate_binary.eval_param_str", return_value="fixed")
    def test_evaluate_writes_results_and_matrices(
        self, mock_eval_param_str, mock_init, mock_load
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                story_name = "story"
                sub_id = "sub"
                story_segments = ["s1", "s2"]
                recall_segments = ["r1", "r2"]
                rm_comparison = np.array([[1, 0], [0, 1]])

                mock_load.return_value = (
                    [(story_name, sub_id, story_segments, recall_segments)],
                    {story_name: {sub_id: rm_comparison}},
                )

                class DummyRater(Rater):
                    def __init__(self):
                        super().__init__()
                        self.rater_name = "dummy"
                        self.model_name = None

                    def compute_ratings_single_sub(
                        self,
                        story_segments: list[str],
                        recall_segments: list[str],
                        output_scores: bool = False,
                    ) -> list[tuple[int, list[int] | list[tuple[int, float]]]]:
                        return [(0, [0]), (1, [1])]

                mock_init.return_value = DummyRater()

                evaluate(
                    repeat_reliability=False,
                    rater_name="openai",
                    model_name=None,
                    testset="cyoa_alice10",
                    device=None,
                    seed=0,
                    random_mode=None,
                )

                results_path = Path("data") / "eval" / "fixed" / "results.json"
                self.assertTrue(results_path.exists())
                results = json.loads(results_path.read_text())
                self.assertAlmostEqual(results["f1_macro"], 1.0)
                self.assertAlmostEqual(results["accuracy_macro"], 1.0)

                self.assertTrue(
                    (
                        Path("data") / "eval" / "fixed" / "recall_matrices_model.pkl"
                    ).exists()
                )
                self.assertTrue(
                    (
                        Path("data")
                        / "eval"
                        / "fixed"
                        / "recall_matrices_cyoa_alice10.pkl"
                    ).exists()
                )
            finally:
                os.chdir(cwd)

    @patch("recall_matrix.evaluate_binary.load_story_recall_segments_default")
    @patch("recall_matrix.evaluate_binary.initialize_rater")
    @patch("recall_matrix.evaluate_binary.eval_param_str", return_value="fixed")
    def test_evaluate_raises_if_all_zero(
        self, mock_eval_param_str, mock_init, mock_load
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                story_name = "story"
                sub_id = "sub"
                story_segments = ["s1", "s2"]
                recall_segments = ["r1", "r2"]
                rm_comparison = np.zeros((2, 2), dtype=int)

                mock_load.return_value = (
                    [(story_name, sub_id, story_segments, recall_segments)],
                    {story_name: {sub_id: rm_comparison}},
                )

                class DummyRater(Rater):
                    def __init__(self):
                        super().__init__()
                        self.rater_name = "dummy"
                        self.model_name = None

                    def compute_ratings_single_sub(
                        self,
                        story_segments: list[str],
                        recall_segments: list[str],
                        output_scores: bool = False,
                    ) -> list[tuple[int, list[int] | list[tuple[int, float]]]]:
                        return [(0, [0]), (1, [1])]

                mock_init.return_value = DummyRater()

                with self.assertRaises(ValueError):
                    evaluate(
                        repeat_reliability=False,
                        rater_name="openai",
                        model_name=None,
                        testset="cyoa_alice10",
                        device=None,
                        seed=0,
                        random_mode=None,
                    )
            finally:
                os.chdir(cwd)

    def test_get_average_pairwise_f1_and_krippendorff_alpha(self):
        # create 2 recall matrices for one story
        m1 = np.array([[1, 0], [0, 1]])
        m2 = np.array([[1, 0], [0, 1]])
        d = {"s1": [m1, m2]}

        # average pairwise F1 should be 1.0 when matrices are identical
        self.assertAlmostEqual(get_average_pairwise_f1(d), 1.0)

        # krippendorff should be 1.0 for identical matrices as well
        # depending on krippendorff implementation, exact float may vary slightly
        alpha = get_krippendorff_alpha(d)
        self.assertGreaterEqual(alpha, 0.999)


if __name__ == "__main__":
    unittest.main()
