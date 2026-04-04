import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from rmatch import get_logger

log = get_logger(__name__)


def get_param_str(config_dict: dict) -> str:
    """Get the param string from the output dict."""

    param_str_ls = list()
    matcher_name = config_dict["matcher_name"]
    param_str_ls.append(matcher_name)
    recall_segmentation = config_dict["recall_segmentation"]
    param_str_ls.append(recall_segmentation)
    story_segmentation = config_dict["story_segmentation"]
    param_str_ls.append(story_segmentation)

    return "-".join(param_str_ls)


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


def save_to_json(out_path: Path, output_dict: dict):
    """Sanitize and save the output dict to a json file."""

    output_dict = deepcopy(output_dict)

    # convert potential numpy values into python native values
    # (json cannot handle numpy values)
    matches_converted = dict()
    for sub_id, single_sub_matches in output_dict["matches"].items():
        matches_converted[sub_id] = list()
        for recall_segment_id, story_segment_ids in single_sub_matches:
            story_segment_ids_converted = list()
            for story_segment_id in story_segment_ids:
                if isinstance(story_segment_id, np.generic):
                    story_segment_id = story_segment_id.item()

                story_segment_ids_converted.append(story_segment_id)

            matches_converted[sub_id].append(
                (recall_segment_id, story_segment_ids_converted)
            )
    output_dict["matches"] = matches_converted

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f_out:
        f_out.write(json.dumps(output_dict) + "\n")
    log.info(f"Saved matches to {out_path}")

    return out_path
