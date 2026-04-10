import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rmatch import get_logger, matchlist_type

log = get_logger(__name__)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to ``path`` atomically (temp file then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(text, encoding=encoding)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_json(
    path: Path, obj: Any, *, indent: int | None = None, default: Any = None
) -> None:
    """Serialize ``obj`` to JSON and write atomically."""
    if default is None:

        def default_fn(o: Any) -> Any:
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(
                f"Object of type {type(o).__name__} is not JSON serializable"
            )

    else:
        default_fn = default  # type: ignore[assignment]

    text = json.dumps(obj, indent=indent, default=default_fn) + "\n"
    atomic_write_text(path, text)


def get_param_str(config_dict: dict) -> str:
    """Build a filename-safe param string from matcher + segmentation config."""
    m = config_dict["matcher_name"]
    r = config_dict["recall_segmentation"]
    s = config_dict["story_segmentation"]
    return f"{m}-{r}-{s}"


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return np.nan

    xm = x - x.mean()
    ym = y - y.mean()
    ss_xx = float(np.dot(xm, xm))
    ss_yy = float(np.dot(ym, ym))
    if ss_xx == 0.0 or ss_yy == 0.0:
        return np.nan

    r = float(np.dot(xm, ym) / np.sqrt(ss_xx * ss_yy))
    r = max(-1.0, min(1.0, r))

    return r


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision for binary labels: TP / (TP + FP)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for binary labels: TP / (TP + FN)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for binary labels: 2*P*R / (P + R)."""
    p = binary_precision(y_true, y_pred)
    r = binary_recall(y_true, y_pred)
    denom = p + r
    return (2 * p * r) / denom if denom > 0 else 0.0


def match_list_to_matrix(
    match_list: matchlist_type, n_story_segments: int
) -> np.ndarray:
    """Convert the ratings to a recall matrix.

    Returns
    -------
    recall_matrix: np.ndarray
        recall matrix of shape (n_story_segments, n_recall_segments)
    """
    n_recall_segments = len(match_list)
    recall_matrix = np.zeros((n_story_segments, n_recall_segments), dtype=int)

    for idx_recall_segment, story_segment_indices in match_list:
        for idx_story_segment in story_segment_indices:
            recall_matrix[idx_story_segment, idx_recall_segment] = 1
    return recall_matrix
