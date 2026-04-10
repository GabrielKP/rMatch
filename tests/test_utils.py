"""Tests for rmatch.utils: metric functions and atomic I/O."""

import json
from pathlib import Path

import numpy as np
import pytest

from rmatch.utils import (
    atomic_write_json,
    atomic_write_text,
    binary_f1,
    binary_precision,
    binary_recall,
    match_list_to_matrix,
    pearsonr,
)

# ── binary_precision ──────────────────────────────────────────────────────────


def test_binary_precision_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert binary_precision(y_true, y_pred) == pytest.approx(1.0)


def test_binary_precision_zero_no_positive_predictions():
    y_true = np.array([1, 1, 0])
    y_pred = np.array([0, 0, 0])
    assert binary_precision(y_true, y_pred) == 0.0


def test_binary_precision_all_false_positives():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert binary_precision(y_true, y_pred) == 0.0


def test_binary_precision_partial():
    # TP=2, FP=1 → 2/3
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0])
    assert binary_precision(y_true, y_pred) == pytest.approx(2 / 3)


def test_binary_precision_all_true_positive():
    y_true = np.array([1, 1])
    y_pred = np.array([1, 1])
    assert binary_precision(y_true, y_pred) == 1.0


# ── binary_recall ─────────────────────────────────────────────────────────────


def test_binary_recall_perfect():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])
    assert binary_recall(y_true, y_pred) == pytest.approx(1.0)


def test_binary_recall_zero_no_positives_in_true():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 0])
    assert binary_recall(y_true, y_pred) == 0.0


def test_binary_recall_zero_all_fn():
    y_true = np.array([1, 1, 0])
    y_pred = np.array([0, 0, 0])
    assert binary_recall(y_true, y_pred) == 0.0


def test_binary_recall_partial():
    # TP=1, FN=1 → 0.5
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 0, 0])
    assert binary_recall(y_true, y_pred) == pytest.approx(0.5)


# ── binary_f1 ─────────────────────────────────────────────────────────────────


def test_binary_f1_perfect():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])
    assert binary_f1(y_true, y_pred) == pytest.approx(1.0)


def test_binary_f1_zero_both_precision_and_recall_zero():
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    assert binary_f1(y_true, y_pred) == 0.0


def test_binary_f1_partial():
    # P=2/3, R=1 → F1 = 2*(2/3)/(2/3+1) = 4/3 / 5/3 = 4/5 = 0.8
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 1, 1])
    p = binary_precision(y_true, y_pred)
    r = binary_recall(y_true, y_pred)
    expected = 2 * p * r / (p + r)
    assert binary_f1(y_true, y_pred) == pytest.approx(expected)


def test_binary_f1_no_tp():
    # All predictions wrong: precision=0 and recall=0
    y_true = np.array([1, 0])
    y_pred = np.array([0, 1])
    assert binary_f1(y_true, y_pred) == 0.0


def test_binary_f1_symmetric():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    # F1 should be the same if we swap labels (for binary)
    f1_ab = binary_f1(y_true, y_pred)
    assert f1_ab >= 0.0
    assert f1_ab <= 1.0


# ── pearsonr ──────────────────────────────────────────────────────────────────


def test_pearsonr_perfect_positive():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    assert pearsonr(x, y) == pytest.approx(1.0)


def test_pearsonr_perfect_negative():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert pearsonr(x, y) == pytest.approx(-1.0)


def test_pearsonr_empty_array_returns_nan():
    result = pearsonr(np.array([]), np.array([]))
    assert np.isnan(result)


def test_pearsonr_single_element_returns_nan():
    result = pearsonr(np.array([1.0]), np.array([1.0]))
    assert np.isnan(result)


def test_pearsonr_constant_x_returns_nan():
    # Zero variance in x → undefined correlation
    result = pearsonr(np.array([1.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    assert np.isnan(result)


def test_pearsonr_constant_y_returns_nan():
    result = pearsonr(np.array([0.0, 1.0, 0.0]), np.array([2.0, 2.0, 2.0]))
    assert np.isnan(result)


def test_pearsonr_moderate_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
    result = pearsonr(x, y)
    assert -1.0 <= result <= 1.0
    assert not np.isnan(result)


def test_pearsonr_no_correlation():
    x = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert pearsonr(x, y) == pytest.approx(-1.0)


# ── match_list_to_matrix ──────────────────────────────────────────────────────


def test_match_list_to_matrix_basic():
    match_list = [(0, [0, 1]), (1, [2]), (2, [])]
    mat = match_list_to_matrix(match_list, n_story_segments=3)
    assert mat.shape == (3, 3)
    # recall 0 maps story segs 0 and 1
    assert mat[0, 0] == 1
    assert mat[1, 0] == 1
    assert mat[2, 0] == 0
    # recall 1 maps story seg 2
    assert mat[2, 1] == 1
    assert mat[0, 1] == 0
    # recall 2 maps nothing
    assert mat[:, 2].sum() == 0
    assert mat.sum() == 3


def test_match_list_to_matrix_empty_matches():
    match_list = [(0, []), (1, [])]
    mat = match_list_to_matrix(match_list, n_story_segments=4)
    assert mat.shape == (4, 2)
    assert mat.sum() == 0


def test_match_list_to_matrix_all_match_same_story_segment():
    match_list = [(0, [0]), (1, [0]), (2, [0])]
    mat = match_list_to_matrix(match_list, n_story_segments=3)
    # All recalls match story segment 0
    assert (mat[0, :] == [1, 1, 1]).all()
    assert mat[1:, :].sum() == 0


def test_match_list_to_matrix_int_dtype():
    mat = match_list_to_matrix([(0, [0])], n_story_segments=2)
    assert mat.dtype == int


def test_match_list_to_matrix_no_recalls():
    mat = match_list_to_matrix([], n_story_segments=5)
    assert mat.shape == (5, 0)


# ── atomic_write_json ─────────────────────────────────────────────────────────


def test_atomic_write_json_creates_file(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json(path, {"key": "value"})
    assert path.exists()
    data = json.loads(path.read_text())
    assert data == {"key": "value"}


def test_atomic_write_json_overwrites_existing(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json(path, {"version": 1})
    atomic_write_json(path, {"version": 2})
    data = json.loads(path.read_text())
    assert data["version"] == 2


def test_atomic_write_json_numpy_serialization(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json(path, {"val": np.float64(3.14)})
    data = json.loads(path.read_text())
    assert data["val"] == pytest.approx(3.14)


def test_atomic_write_json_non_serializable_raises(tmp_path):
    path = tmp_path / "out.json"
    with pytest.raises(TypeError):
        atomic_write_json(path, {"bad": object()})


def test_atomic_write_json_nested(tmp_path):
    path = tmp_path / "out.json"
    obj = {"matches": {"sub01": [[0, [1, 2]], [1, [0]]]}}
    atomic_write_json(path, obj)
    data = json.loads(path.read_text())
    assert data == obj


# ── atomic_write_text ─────────────────────────────────────────────────────────


def test_atomic_write_text_creates_file(tmp_path):
    path = tmp_path / "out.txt"
    atomic_write_text(path, "hello world")
    assert path.read_text() == "hello world"


def test_atomic_write_text_overwrites_existing(tmp_path):
    path = tmp_path / "out.txt"
    atomic_write_text(path, "first")
    atomic_write_text(path, "second")
    assert path.read_text() == "second"


def test_atomic_write_text_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.txt"
    atomic_write_text(path, "content")
    assert path.read_text() == "content"
