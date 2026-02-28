from pathlib import Path

import numpy as np

from recall_matrix.load import load_story_recall_segments
from recall_matrix.raters import initialize_rater
from recall_matrix.utils import (
    human_csv_to_matrix,
    inspect_recall_segments,
    plot_recall_matrix,
    ratings_to_matrix,
    threshold_human_matrix,
)

STORY = "pieman"
SUBJECT = "sub-003"


if __name__ == "__main__":
    rater = initialize_rater(
        rater_name="anthropic",
        model_name="claude-opus-4-6",
        window_size=5,
    )

    output = rater.rate(
        story_name=STORY,
        story_segmentation_method="sentences_corrected",
        recall_segmentation_method="sentences",
        sub_ids=[SUBJECT],
    )

    llm_matrix = ratings_to_matrix(output, SUBJECT).astype(float)

    human_csv_path = Path(
        f"data/stories-and-recalls/{STORY}/recall_quality/human/"
        f"{SUBJECT}-DA-ssm_sentences_corrected-rsm_sentences.csv"
    )

    human_matrix_norm = human_csv_to_matrix(human_csv_path)
    human_matrix_bin = threshold_human_matrix(human_matrix_norm)

    diff = np.abs(human_matrix_bin - llm_matrix)

    disagreement_mask = diff.sum(axis=0) > 0
    disagreed_indices = np.where(disagreement_mask)[0].tolist()

    data, _, _ = load_story_recall_segments(
        story_name=STORY,
        story_segmentation_method="sentences_corrected",
        recall_segmentation_method="sentences",
        sub_ids=[SUBJECT],
    )

    _, story_segments, recall_segments = data[0]

    inspect_recall_segments(
        recall_segments=recall_segments,
        story_segments=story_segments,
        human_matrix=human_matrix_norm,
        llm_matrix=llm_matrix,
        recall_indices=disagreed_indices,
    )

    plot_recall_matrix(
        STORY,
        SUBJECT,
        llm_matrix,
        "LLM Binary Ratings",
        Path("outputs/tests/anthropic/llm"),
    )

    plot_recall_matrix(
        STORY,
        f"{SUBJECT}-diff",
        diff,
        "Difference (Human vs LLM)",
        Path("outputs/tests/anthropic/comparison"),
        measure_label="|human - model|",
    )

    plot_recall_matrix(
        story_name=STORY,
        sub_id=f"{SUBJECT}-human",
        recall_matrix=human_matrix_norm,
        title="Human Ratings (Normalized)",
        output_dir=Path("outputs/tests/anthropic/human"),
        measure_label="normalized score (0-1)",
    )

    print("THIS IS A TEST OF THE COST TRACKING")
    print(rater.get_usage())
