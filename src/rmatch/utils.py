import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rmatch import get_logger

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


def ratings_single_sub_to_matrix(
    ratings_single_sub: list[tuple[int, list[int]]], n_story_segments: int
) -> np.ndarray:
    """Convert the ratings to a recall matrix.

    Returns
    -------
    recall_matrix: np.ndarray
        recall matrix of shape (n_story_segments, n_recall_segments)
    """
    n_recall_segments = len(ratings_single_sub)
    recall_matrix = np.zeros((n_story_segments, n_recall_segments), dtype=int)

    for idx_recall_segment, story_segment_indices in ratings_single_sub:
        for idx_story_segment in story_segment_indices:
            recall_matrix[idx_story_segment, idx_recall_segment] = 1
    return recall_matrix
