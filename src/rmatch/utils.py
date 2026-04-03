import numpy as np


def get_param_str(output_dict: dict) -> str:
    """Get the param string from the output dict."""

    matcher_name = output_dict["matcher_name"]
    recall_segmentation_method = output_dict["recall_segmentation_method"]
    story_segmentation_method = output_dict["story_segmentation_method"]
    output_scores_str = "-ouput_scores" if output_dict["output_scores"] else ""

    param_str = (
        f"{matcher_name}"
        f"-{story_segmentation_method}"
        f"-{recall_segmentation_method}"
        f"{output_scores_str}"
    )
    return param_str


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
