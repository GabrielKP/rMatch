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

    def test_load_story_recall_segments_default_invalid_testset(self):
        """Test that invalid testset raises ValueError."""
        from recall_matrix.evaluate_binary import load_story_recall_segments_default

        with self.assertRaises(ValueError) as context:
            load_story_recall_segments_default("invalid_testset")

        self.assertIn("Invalid testset", str(context.exception))

    @patch("recall_matrix.evaluate_binary.load_cyoa_recall_matrix_human_binary")
    @patch("recall_matrix.evaluate_binary.load_cyoa_story_recall_segments")
    def test_load_story_recall_segments_default_cyoa_default(
        self, mock_load_cyoa, mock_load_human
    ):
        """Test loading cyoa_alice10 testset."""
        from recall_matrix.evaluate_binary import load_story_recall_segments_default

        mock_load_cyoa.return_value = [
            ("alice_2", "sub-001", ["s1"], ["r1"]),
        ]
        mock_load_human.return_value = np.array([[1, 0]])

        story_recall_segments, human_ratings = load_story_recall_segments_default(
            "cyoa_alice10"
        )

        self.assertEqual(len(story_recall_segments), 1)
        self.assertIsNotNone(human_ratings)
        self.assertIn("alice_2", human_ratings)  # type: ignore
        self.assertIn("sub-001", human_ratings["alice_2"])  # type: ignore
        mock_load_cyoa.assert_called_once()

    def test_accuracy_with_data(self):
        """Test accuracy function with real data."""
        from recall_matrix.evaluate_binary import accuracy

        arr1 = np.array([1, 0, 1, 1, 0])
        arr2 = np.array([1, 0, 1, 0, 0])

        acc = accuracy(arr1, arr2)
        self.assertAlmostEqual(acc, 0.8)  # 4 out of 5 correct


class TestRatersWithNumpyValues(unittest.TestCase):
    """Test rater JSON serialization with numpy values."""

    def test_save_to_json_with_numpy_values_and_output_scores(self):
        """Test save_to_json converts numpy values correctly when output_scores=True."""
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
                    "n_story_segments": 2,
                    "output_scores": True,
                    "ratings": {
                        "sub-001": [
                            (
                                np.int64(0),
                                [
                                    (np.int64(0), np.float32(0.8)),
                                    (np.int64(1), np.float64(0.5)),
                                ],
                            )
                        ]
                    },
                }

                output_path = rater.save_to_json(output_dict)

                # Load and verify
                full_path = os.path.join(tmpdir, str(output_path))
                with open(full_path, "r") as f:
                    loaded = json.load(f)

                # Verify numpy types were converted to Python native types
                ratings = loaded["ratings"]["sub-001"]
                self.assertIsInstance(ratings[0][0], int)
                self.assertIsInstance(ratings[0][1][0][0], int)
                self.assertIsInstance(ratings[0][1][0][1], float)

            finally:
                os.chdir(cwd)

    def test_save_to_json_with_numpy_values_no_output_scores(self):
        """Test save_to_json converts numpy values when output_scores=False."""
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
                    "n_story_segments": 2,
                    "output_scores": False,
                    "ratings": {
                        "sub-001": [
                            (np.int64(0), [np.int64(0), np.int64(1)]),
                        ]
                    },
                }

                output_path = rater.save_to_json(output_dict)

                # Load and verify
                full_path = os.path.join(tmpdir, str(output_path))
                with open(full_path, "r") as f:
                    loaded = json.load(f)

                # Verify numpy types were converted
                ratings = loaded["ratings"]["sub-001"]
                self.assertIsInstance(ratings[0][0], int)
                self.assertIsInstance(ratings[0][1][0], int)

            finally:
                os.chdir(cwd)


class TestRaterOpenAIParse(unittest.TestCase):
    """Test RaterOpenAI parse method edge cases."""

    def test_parse_invalid_format_logs_warning(self):
        """Test parse method with invalid format logs warning."""
        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")

            # This should return empty set and log a warning
            result = rater.parse("invalid format without brackets")
            self.assertEqual(result, set())

    def test_parse_with_none(self):
        """Test parse method with <NONE>."""
        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")
            result = rater.parse("<NONE>")
            self.assertEqual(result, set())

    def test_parse_with_trailing_whitespace(self):
        """Test parse method with trailing whitespace."""
        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")
            result = rater.parse("  <1, 2, 3>  ")
            self.assertEqual(result, {1, 2, 3})

    def test_parse_with_single_number(self):
        """Test parse method with single number."""
        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")
            result = rater.parse("<5>")
            self.assertEqual(result, {5})

    def test_parse_with_non_digit_characters(self):
        """Test parse method ignores non-digit characters."""
        with patch.dict("recall_matrix.ENV", {"OPENAI_API_KEY": "key"}, clear=True):
            rater = RaterOpenAI(model_name="gpt-test")
            result = rater.parse("<1, abc, 2>")
            # Should only parse digits
            self.assertEqual(result, {1, 2})


class TestRerankerEdgeCases(unittest.TestCase):
    """Test RaterReranker edge cases."""

    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_reranker_all_scores_below_threshold(self, mock_crossencoder):
        """Test reranker when all scores are below threshold."""
        mock_model = Mock()
        mock_model.rank.return_value = [
            {"score": 0.3, "corpus_id": 0},
            {"score": 0.2, "corpus_id": 1},
        ]
        mock_crossencoder.return_value = mock_model

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            rater = RaterReranker(
                model_name="model-x", device="cpu", threshold=0.5, top_k=5
            )
            ratings = rater.compute_ratings_single_sub(
                story_segments=["s1", "s2"],
                recall_segments=["r1"],
                output_scores=False,
            )

        # Should return empty list because no scores meet threshold
        self.assertEqual(ratings, [(0, [])])

    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_reranker_with_threshold_and_top_k_combined(self, mock_crossencoder):
        """Test reranker respects both threshold and top_k."""
        mock_model = Mock()
        mock_model.rank.return_value = [
            {"score": 0.9, "corpus_id": 0},  # Above threshold and within top_k
            {"score": 0.8, "corpus_id": 1},  # Above threshold and within top_k
            {"score": 0.7, "corpus_id": 2},  # Above threshold but outside top_k
            {"score": 0.3, "corpus_id": 3},  # Below threshold
        ]
        mock_crossencoder.return_value = mock_model

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            rater = RaterReranker(
                model_name="model-x", device="cpu", threshold=0.6, top_k=2
            )
            ratings = rater.compute_ratings_single_sub(
                story_segments=["s1", "s2", "s3", "s4"],
                recall_segments=["r1"],
                output_scores=True,
            )

        # Should include 0.9 and 0.8 (top_k=2, both above threshold)
        self.assertEqual(len(ratings[0][1]), 2)
        self.assertEqual(ratings[0][1][0], (0, 0.9))
        self.assertEqual(ratings[0][1][1], (1, 0.8))


class TestPlotFunctions(unittest.TestCase):
    """Test plot_single_sub functions with mocking."""

    @patch("recall_matrix.plot_single_sub.plt.savefig")
    @patch("recall_matrix.plot_single_sub.plt.close")
    def test_plot_single_sub_recall_matrices(self, mock_close, mock_savefig):
        """Test plot_single_sub_recall_matrices creates output."""
        from recall_matrix.plot_single_sub import plot_single_sub_recall_matrices

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                recall_matrices = [
                    np.array([[1, 0], [0, 1]]),
                    np.array([[1, 1], [0, 0]]),
                ]
                rater_names = ["rater1", "rater2"]

                plot_single_sub_recall_matrices(
                    story_name="test_story",
                    sub_id="sub-001",
                    story_segmentation_method="sentences",
                    recall_segmentation_method="sentences",
                    recall_matrices=recall_matrices,
                    rater_names=rater_names,
                )

                # Verify that savefig was called
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()

                # Verify output directory structure
                plots_dir = (
                    Path("data") / "stories-and-recalls" / "test_story" / "plots"
                )
                self.assertTrue(plots_dir.exists())

            finally:
                os.chdir(cwd)

    @patch("recall_matrix.plot_single_sub.plot_single_sub_recall_matrices")
    def test_plot_single_sub_loads_ratings_from_json(self, mock_plot):
        """Test plot_single_sub loads ratings from JSON files."""
        from recall_matrix.plot_single_sub import plot_single_sub

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSON file
            test_json = {
                "rater_name": "test_rater",
                "story_name": "test_story",
                "story_segmentation_method": "sentences",
                "recall_segmentation_method": "sentences",
                "n_story_segments": 2,
                "output_scores": False,
                "ratings": {
                    "sub-001": [(0, [0, 1]), (1, [1])],
                },
            }

            json_file = Path(tmpdir) / "test_ratings.json"
            json_file.write_text(json.dumps(test_json))

            plot_single_sub(
                sub_id="sub-001",
                paths_ratings=[json_file],
            )

            # Verify plot function was called
            mock_plot.assert_called_once()


class TestRaterInitialize(unittest.TestCase):
    """Test rater initialization edge cases."""

    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_initialize_reranker_without_model_name(self, mock_ce):
        """Test initializing reranker without explicit model name."""
        mock_ce.return_value = Mock()

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            rater = initialize_rater(
                rater_name="reranker",
                model_name=None,
                device="cpu",
            )

        self.assertEqual(rater.rater_name, "reranker")
        # Reranker should have model name attribute
        self.assertTrue(hasattr(rater, "model_name"))


class TestRaterHuggingFace(unittest.TestCase):
    """Test RaterHuggingFace class."""

    def test_rater_huggingface_init(self):
        """Test HuggingFace rater initialization."""
        from recall_matrix.raters.rater_huggingface import RaterHuggingFace

        # Test with default model name
        rater1 = RaterHuggingFace()
        self.assertEqual(rater1.rater_name, "openai")

        # Test with explicit model name
        rater2 = RaterHuggingFace(model_name="bert-base")
        self.assertEqual(rater2.rater_name, "openai")

    def test_rater_huggingface_not_implemented(self):
        """Test HuggingFace rater raises NotImplementedError."""
        from recall_matrix.raters.rater_huggingface import RaterHuggingFace

        rater = RaterHuggingFace()
        with self.assertRaises(NotImplementedError):
            rater.single_subject_ratings(["story"], ["recall"])


class TestPlotSingleRecallMatrix(unittest.TestCase):
    """Test plotting with single recall matrix."""

    @patch("recall_matrix.plot_single_sub.plt.savefig")
    @patch("recall_matrix.plot_single_sub.plt.close")
    def test_plot_single_sub_recall_matrices_single_matrix(
        self, mock_close, mock_savefig
    ):
        """Test plot_single_sub_recall_matrices with single recall matrix."""
        from recall_matrix.plot_single_sub import plot_single_sub_recall_matrices

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Single recall matrix - this triggers the axes = [axes] branch
                recall_matrices = [np.array([[1, 0], [0, 1]])]
                rater_names = ["single_rater"]

                plot_single_sub_recall_matrices(
                    story_name="test_story",
                    sub_id="sub-001",
                    story_segmentation_method="sentences",
                    recall_segmentation_method="sentences",
                    recall_matrices=recall_matrices,
                    rater_names=rater_names,
                )

                # Verify that savefig was called
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()

            finally:
                os.chdir(cwd)


class TestRerankerDefaultThreshold(unittest.TestCase):
    """Test RaterReranker with default threshold and top_k."""

    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_reranker_uses_instance_threshold_when_none_passed(self, mock_crossencoder):
        """Test reranker uses instance threshold when None is passed."""
        mock_model = Mock()
        mock_model.rank.return_value = [
            {"score": 0.5, "corpus_id": 0},
        ]
        mock_crossencoder.return_value = mock_model

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            rater = RaterReranker(
                model_name="model-x", device="cpu", threshold=0.4, top_k=3
            )
            # Call with threshold=None to trigger default logic
            ratings = rater.compute_ratings_single_sub(
                story_segments=["s1"],
                recall_segments=["r1"],
                output_scores=False,
                threshold=None,
                top_k=None,
            )

        # Score 0.5 is above instance threshold 0.4
        self.assertEqual(ratings, [(0, [0])])

    @patch("recall_matrix.raters.rater_reranker.CrossEncoder")
    def test_reranker_uses_default_values_when_instance_none(self, mock_crossencoder):
        """Test reranker falls back to defaults (0.09, 5) when no threshold/top_k."""
        mock_model = Mock()
        mock_model.rank.return_value = [
            {"score": 0.1, "corpus_id": 0},
        ]
        mock_crossencoder.return_value = mock_model

        with patch.dict("recall_matrix.ENV", {"HF_TOKEN": "token"}, clear=True):
            # Initialize with None values
            rater = RaterReranker(model_name="model-x", device="cpu")
            ratings = rater.compute_ratings_single_sub(
                story_segments=["s1"],
                recall_segments=["r1"],
                output_scores=False,
                threshold=None,
                top_k=None,
            )

        # Score 0.1 is above default threshold 0.09
        self.assertEqual(ratings, [(0, [0])])


class TestRaterWithSubIds(unittest.TestCase):
    """Test rater with explicit sub_ids parameter."""

    @patch("recall_matrix.raters.rater.load_story_recall_segments")
    def test_rater_with_explicit_sub_ids(self, mock_load):
        """Test rater.rate with explicit sub_ids parameter."""
        # When load_story_recall_segments is called with sub_ids, it filters
        # the segments to only those matching the requested sub_ids
        story_recall_segments_filtered = [
            ("sub-001", ["TRANSCRIPT A1", "TRANSCRIPT A2"], ["RECALL A1"]),
            ("sub-003", ["TRANSCRIPT C1"], ["RECALL C1"]),
        ]
        mock_load.return_value = (
            story_recall_segments_filtered,
            "sentences",
            "sentences",
        )

        dummy = DummyRaterForTests()
        output = dummy.rate(
            story_name="dummy_story",
            story_segmentation_method="sentences",
            recall_segmentation_method="sentences",
            sub_ids=["sub-001", "sub-003"],  # Only these 2
            output_scores=False,
        )

        # Should only be called for the filtered sub_ids
        self.assertEqual(len(dummy.calls), 2)
        self.assertIn("sub-001", output["ratings"])
        self.assertIn("sub-003", output["ratings"])
        # sub-002 should not be present
        self.assertNotIn("sub-002", output["ratings"])


class TestRaterJsonSerializationAllCases(unittest.TestCase):
    """Test rater save_to_json with all numpy type combinations."""

    def test_save_to_json_mixed_numpy_types_with_scores(self):
        """Test save_to_json with mixed numpy types in output_scores mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                rater = Rater()
                rater.rater_name = "test"

                output_dict = {
                    "rater_name": "test",
                    "story_name": "test_story",
                    "story_segmentation_method": "sentences",
                    "recall_segmentation_method": "sentences",
                    "n_story_segments": 3,
                    "output_scores": True,
                    "ratings": {
                        "sub-001": [
                            (
                                np.int32(0),
                                [
                                    (np.int16(0), np.float16(0.9)),
                                    (1, np.float32(0.7)),
                                ],
                            ),
                            (1, [(np.int64(1), 0.5)]),
                        ],
                    },
                }

                output_path = rater.save_to_json(output_dict)
                full_path = os.path.join(tmpdir, str(output_path))

                with open(full_path, "r") as f:
                    loaded = json.load(f)

                # Verify all types were converted
                ratings = loaded["ratings"]["sub-001"]
                self.assertIsInstance(ratings[0][0], int)
                self.assertIsInstance(ratings[0][1][0][0], int)
                self.assertIsInstance(ratings[0][1][0][1], float)
                self.assertIsInstance(ratings[0][1][1][0], int)
                self.assertIsInstance(ratings[0][1][1][1], float)

            finally:
                os.chdir(cwd)


class TestEvalParamStr(unittest.TestCase):
    """Test eval_param_str with different parameter combinations."""

    def test_eval_param_str_with_model_name_none(self):
        """Test eval_param_str when model_name is None (line 41->43)."""
        from recall_matrix.evaluate_binary import eval_param_str

        with patch(
            "recall_matrix.evaluate_binary.datetime.datetime", wraps=datetime.datetime
        ) as mock_dt:
            mock_dt.now.return_value = datetime.datetime(2020, 1, 1, 0, 0, 0)
            param_str = eval_param_str(
                repeat_reliability=False,
                testset="cyoa_alice10",
                rater_name="test_rater",
                model_name=None,  # This is the key - None value
                seed=42,
                random_mode=None,
            )

        self.assertIn("20200101_000000", param_str)
        self.assertIn("cyoa_alice10", param_str)
        self.assertIn("test_rater", param_str)
        self.assertNotIn("-m_", param_str)  # No model name part

    def test_eval_param_str_with_repeat_reliability(self):
        """Test eval_param_str when repeat_reliability is True."""
        from recall_matrix.evaluate_binary import eval_param_str

        with patch(
            "recall_matrix.evaluate_binary.datetime.datetime", wraps=datetime.datetime
        ) as mock_dt:
            mock_dt.now.return_value = datetime.datetime(2020, 1, 1, 0, 0, 0)
            param_str = eval_param_str(
                repeat_reliability=True,
                testset="cyoa_alice10",
                rater_name="test_rater",
                model_name=None,
                seed=42,
                random_mode=None,
            )

        self.assertIn("-rr", param_str)


class TestLoadStoryRecallSegmentsMonthiversary(unittest.TestCase):
    """Test load_story_recall_segments_default with cyoa_monthiversary6."""

    @patch("recall_matrix.evaluate_binary.load_cyoa_recall_matrix_human_binary")
    @patch("recall_matrix.evaluate_binary.load_cyoa_story_recall_segments")
    def test_load_cyoa_monthiversary6(self, mock_load_cyoa, mock_load_human):
        """Test that cyoa_monthiversary6 testset works (lines 79-90)."""
        from recall_matrix.evaluate_binary import load_story_recall_segments_default

        mock_load_cyoa.return_value = [
            ("monthiversary_3", "sub-001", ["s1"], ["r1"]),
        ]
        mock_load_human.return_value = np.array([[1, 0]])

        story_recall_segments, human_ratings = load_story_recall_segments_default(
            "cyoa_monthiversary6"
        )

        self.assertEqual(len(story_recall_segments), 1)
        self.assertIsNotNone(human_ratings)
        mock_load_cyoa.assert_called_once()


class TestLoadStoryRecallSegmentsMemsearch(unittest.TestCase):
    """Test load_story_recall_segments_default with memsearch testsets."""

    @patch("recall_matrix.evaluate_binary.load_ratings_dict")
    @patch("recall_matrix.evaluate_binary.ratings_single_sub_to_matrix")
    @patch("recall_matrix.evaluate_binary.load_story_recall_segments")
    def test_load_memsearch10(self, mock_load, mock_to_matrix, mock_ratings_dict):
        """Test that memsearch10 testset loads correctly (lines 102-169)."""
        from recall_matrix.evaluate_binary import load_story_recall_segments_default

        # Mock load_story_recall_segments to return one story's segments
        mock_load.return_value = (
            [("sub-001", ["s1"], ["r1"])],
            "seg_c",
            "sentences",
        )

        # Mock ratings dict and conversion
        mock_ratings_dict.return_value = {
            "n_story_segments": 2,
            "ratings": {"sub-001": [(0, [0, 1])]},
        }
        mock_to_matrix.return_value = np.array([[1, 1]])

        story_recall_segments, human_ratings = load_story_recall_segments_default(
            "memsearch10"
        )

        self.assertIsNotNone(story_recall_segments)
        self.assertIsNotNone(human_ratings)

    @patch("recall_matrix.evaluate_binary.load_ratings_dict")
    @patch("recall_matrix.evaluate_binary.ratings_single_sub_to_matrix")
    @patch("recall_matrix.evaluate_binary.load_story_recall_segments")
    def test_load_memsearch_all(self, mock_load, mock_to_matrix, mock_ratings_dict):
        """Test that memsearch (all stories) testset loads correctly."""
        from recall_matrix.evaluate_binary import load_story_recall_segments_default

        mock_load.return_value = (
            [("sub-001", ["s1"], ["r1"])],
            "seg_c",
            "sentences",
        )
        mock_ratings_dict.return_value = {
            "n_story_segments": 2,
            "ratings": {"sub-001": [(0, [0, 1])]},
        }
        mock_to_matrix.return_value = np.array([[1, 1]])

        story_recall_segments, human_ratings = load_story_recall_segments_default(
            "memsearch"
        )

        self.assertIsNotNone(story_recall_segments)


class TestEvaluateWithModelZeroMatrix(unittest.TestCase):
    """Test evaluate function when model produces all-zero matrix."""

    @patch("recall_matrix.evaluate_binary.load_story_recall_segments_default")
    @patch("recall_matrix.evaluate_binary.initialize_rater")
    @patch("recall_matrix.evaluate_binary.eval_param_str", return_value="fixed")
    def test_evaluate_model_all_zeros(self, mock_eval_param_str, mock_init, mock_load):
        """Test evaluate when rm_model is all zeros (lines 284-287)."""
        from recall_matrix.evaluate_binary import evaluate

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                story_name = "story"
                sub_id = "sub"
                story_segments = ["s1", "s2"]
                recall_segments = ["r1"]
                rm_comparison = np.array([[1, 0]])

                mock_load.return_value = (
                    [(story_name, sub_id, story_segments, recall_segments)],
                    {story_name: {sub_id: rm_comparison}},
                )

                class AllZeroRater(Rater):
                    def __init__(self):
                        super().__init__()
                        self.rater_name = "dummy"

                    def compute_ratings_single_sub(  # type: ignore
                        self,
                        story_segments: list[str],
                        recall_segments: list[str],
                        output_scores: bool = False,
                    ):
                        # Return all zeros
                        return [(0, [])]

                mock_init.return_value = AllZeroRater()

                evaluate(
                    repeat_reliability=False,
                    rater_name="test",
                    model_name=None,
                    testset="cyoa_alice10",
                    device=None,
                    seed=42,
                    random_mode=None,
                )

                results_path = Path("data") / "eval" / "fixed" / "results.json"
                self.assertTrue(results_path.exists())

            finally:
                os.chdir(cwd)


class TestEvaluateWithRandomMode(unittest.TestCase):
    """Test evaluate function with random_mode parameter."""

    @patch("recall_matrix.evaluate_binary.load_story_recall_segments_default")
    @patch("recall_matrix.evaluate_binary.initialize_rater")
    @patch("recall_matrix.evaluate_binary.eval_param_str", return_value="fixed")
    def test_evaluate_with_random_mode(self, mock_eval_param_str, mock_init, mock_load):
        """Test evaluate when random_mode is not None (line 303)."""
        from recall_matrix.evaluate_binary import evaluate

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                story_name = "story"
                sub_id = "sub"
                story_segments = ["s1", "s2"]
                recall_segments = ["r1"]
                rm_comparison = np.array([[1, 0]])

                mock_load.return_value = (
                    [(story_name, sub_id, story_segments, recall_segments)],
                    {story_name: {sub_id: rm_comparison}},
                )

                class DummyRater(Rater):
                    def __init__(self):
                        super().__init__()
                        self.rater_name = "dummy"

                    def compute_ratings_single_sub(  # type: ignore
                        self,
                        story_segments: list[str],
                        recall_segments: list[str],
                        output_scores: bool = False,
                    ):
                        return [(0, [0])]

                mock_init.return_value = DummyRater()

                evaluate(
                    repeat_reliability=False,
                    rater_name="test",
                    model_name=None,
                    testset="cyoa_alice10",
                    device=None,
                    seed=42,
                    random_mode="full_shuffle",  # Not None!
                )

                results_path = Path("data") / "eval" / "fixed" / "results.json"
                self.assertTrue(results_path.exists())
                results = json.loads(results_path.read_text())
                self.assertEqual(results["random_mode"], "full_shuffle")

            finally:
                os.chdir(cwd)


class TestLoadStoryRecallSegmentsRepeatReliability(unittest.TestCase):
    """Test load_story_recall_segments_repeat_reliability for different testsets."""

    @patch("recall_matrix.evaluate_binary.load_cyoa_recall_matrix_human_binary")
    @patch("recall_matrix.evaluate_binary.load_cyoa_story_recall_segments")
    def test_load_repeat_reliability_cyoa_alice10(
        self, mock_load_cyoa, mock_load_human
    ):
        """Test with cyoa_alice10 testset (line 372-380)."""
        from recall_matrix.evaluate_binary import (
            load_story_recall_segments_repeat_reliability,
        )

        mock_load_cyoa.return_value = [
            ("alice_3", "sub-001", ["s1"], ["r1"]),
            ("alice_6", "sub-002", ["s2"], ["r2"]),
            ("alice_12", "sub-003", ["s3"], ["r3"]),
        ]
        mock_load_human.return_value = np.array([[1, 0]])

        story_segments, human_ratings = load_story_recall_segments_repeat_reliability(
            "cyoa_alice10"
        )

        self.assertEqual(len(story_segments), 3)
        self.assertIsNotNone(human_ratings)
        self.assertIn("alice_3", human_ratings)  # type: ignore
        self.assertIn("alice_6", human_ratings)  # type: ignore
        self.assertIn("alice_12", human_ratings)  # type: ignore

    @patch("recall_matrix.evaluate_binary.load_cyoa_recall_matrix_human_binary")
    @patch("recall_matrix.evaluate_binary.load_cyoa_story_recall_segments")
    def test_load_repeat_reliability_cyoa_monthiversary6(
        self, mock_load_cyoa, mock_load_human
    ):
        """Test with cyoa_monthiversary6 testset."""
        from recall_matrix.evaluate_binary import (
            load_story_recall_segments_repeat_reliability,
        )

        mock_load_cyoa.return_value = [
            ("monthiversary_3", "sub-001", ["s1"], ["r1"]),
            ("monthiversary_4", "sub-002", ["s2"], ["r2"]),
            ("monthiversary_25", "sub-003", ["s3"], ["r3"]),
        ]
        mock_load_human.return_value = np.array([[1, 0]])

        story_segments, human_ratings = load_story_recall_segments_repeat_reliability(
            "cyoa_monthiversary6"
        )

        self.assertEqual(len(story_segments), 3)

    def test_load_repeat_reliability_invalid_cyoa(self):
        """Test with invalid cyoa testset (line 380-383 raises ValueError)."""
        from recall_matrix.evaluate_binary import (
            load_story_recall_segments_repeat_reliability,
        )

        with self.assertRaises(ValueError) as context:
            load_story_recall_segments_repeat_reliability("cyoa_invalid")

        self.assertIn("Invalid testset", str(context.exception))

    @patch("recall_matrix.evaluate_binary.load_ratings_dict")
    @patch("recall_matrix.evaluate_binary.ratings_single_sub_to_matrix")
    @patch("recall_matrix.evaluate_binary.load_story_recall_segments")
    def test_load_repeat_reliability_memsearch(
        self, mock_load, mock_to_matrix, mock_ratings_dict
    ):
        """Test with memsearch testset (lines 411-447)."""
        from recall_matrix.evaluate_binary import (
            load_story_recall_segments_repeat_reliability,
        )

        # memsearch uses special story names and only picks first recall per story
        mock_load.return_value = (
            [("sub-001", ["s1"], ["r1"])],
            "seg_c",
            "sentences",
        )
        mock_ratings_dict.return_value = {
            "n_story_segments": 2,
            "ratings": {"sub-001": [(0, [0, 1])]},
        }
        mock_to_matrix.return_value = np.array([[1, 1]])

        story_segments, human_ratings = load_story_recall_segments_repeat_reliability(
            "memsearch"
        )

        # Should have 3 specific memsearch stories
        self.assertEqual(len(story_segments), 3)


class TestGetAveragePairwiseF1(unittest.TestCase):
    """Test get_average_pairwise_f1 function edge cases."""

    def test_get_average_pairwise_f1_single_recall_per_story(self):
        """Test get_average_pairwise_f1 with single recall per story."""
        from recall_matrix.evaluate_binary import get_average_pairwise_f1

        # Single matrix per story - no pairs to compare-> should go to line 466
        d = {"s1": [np.array([[1, 0]])], "s2": [np.array([[0, 1]])]}

        f1 = get_average_pairwise_f1(d)
        # Single matrix per story means scores list is empty for each
        # so should return 0.0
        self.assertEqual(f1, 0.0)

    def test_get_average_pairwise_f1_multiple_matrices(self):
        """Test get_average_pairwise_f1 with multiple matrices."""
        from recall_matrix.evaluate_binary import get_average_pairwise_f1

        # Multiple matrices per story - will calculate pairwise F1
        m1 = np.array([[1, 0]])
        m2 = np.array([[1, 0]])
        m3 = np.array([[0, 1]])

        d = {"s1": [m1, m2, m3]}

        f1 = get_average_pairwise_f1(d)
        # Should compute F1 between all pairs
        self.assertIsInstance(f1, float)

    def test_get_average_pairwise_f1_empty_dict(self):
        """Test get_average_pairwise_f1 with empty dictionary."""
        from recall_matrix.evaluate_binary import get_average_pairwise_f1

        d = {}
        f1 = get_average_pairwise_f1(d)
        self.assertEqual(f1, 0.0)


class TestEvaluateRepeatReliability(unittest.TestCase):
    """Test evaluate_repeat_reliability function (lines 505-680)."""

    @patch(
        "recall_matrix.evaluate_binary.load_story_recall_segments_repeat_reliability"
    )
    @patch("recall_matrix.evaluate_binary.initialize_rater")
    @patch("recall_matrix.evaluate_binary.eval_param_str", return_value="fixed")
    def test_evaluate_repeat_reliability_basic(
        self, mock_eval_param_str, mock_init, mock_load
    ):
        """Test evaluate_repeat_reliability basic functionality."""
        from recall_matrix.evaluate_binary import evaluate_repeat_reliability

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                story_name = "story"
                sub_id = "sub"
                story_segments = ["s1", "s2"]
                recall_segments = ["r1"]
                rm_comparison = np.array([[1, 0]])

                mock_load.return_value = (
                    [(story_name, sub_id, story_segments, recall_segments)],
                    {story_name: {sub_id: rm_comparison}},
                )

                class DummyRater(Rater):
                    def __init__(self):
                        super().__init__()
                        self.rater_name = "dummy"

                    def compute_ratings_single_sub(  # type: ignore
                        self,
                        story_segments: list[str],
                        recall_segments: list[str],
                        output_scores: bool = False,
                    ):
                        return [(0, [0])]

                mock_init.return_value = DummyRater()

                evaluate_repeat_reliability(
                    n_repeats=3,
                    rater_name="test",
                    model_name=None,
                    testset="cyoa_alice10",
                    device=None,
                    seed=42,
                    random_mode=None,
                )

                results_path = Path("data") / "eval" / "fixed" / "results.json"
                self.assertTrue(results_path.exists())
                results = json.loads(results_path.read_text())
                self.assertIn("n_repeats", results)
                self.assertEqual(results["n_repeats"], 3)

            finally:
                os.chdir(cwd)

    @patch(
        "recall_matrix.evaluate_binary.load_story_recall_segments_repeat_reliability"
    )
    @patch("recall_matrix.evaluate_binary.initialize_rater")
    def test_evaluate_repeat_reliability_all_zero_comparison(
        self, mock_init, mock_load
    ):
        """Test evaluate_repeat_reliability raises when comparison matrix.

        is all zero.
        """
        from recall_matrix.evaluate_binary import evaluate_repeat_reliability

        story_name = "story"
        sub_id = "sub"
        story_segments = ["s1", "s2"]
        recall_segments = ["r1"]
        rm_comparison = np.zeros((1, 2), dtype=int)

        mock_load.return_value = (
            [(story_name, sub_id, story_segments, recall_segments)],
            {story_name: {sub_id: rm_comparison}},
        )

        class DummyRater(Rater):
            def __init__(self):
                super().__init__()
                self.rater_name = "dummy"

            def compute_ratings_single_sub(  # type: ignore
                self,
                story_segments: list[str],
                recall_segments: list[str],
                output_scores: bool = False,
            ):
                return [(0, [0])]

        mock_init.return_value = DummyRater()

        with self.assertRaises(ValueError) as context:
            evaluate_repeat_reliability(
                n_repeats=3,
                rater_name="test",
                model_name=None,
                testset="cyoa_alice10",
                device=None,
                seed=42,
                random_mode=None,
            )

        self.assertIn("all zero", str(context.exception))


if __name__ == "__main__":
    unittest.main()
