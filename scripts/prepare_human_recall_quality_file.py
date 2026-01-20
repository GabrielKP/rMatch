from pathlib import Path

import pandas as pd

from recall_matrix.load import load_recall_segments, load_story_segments


def generate_file(
    story_name: str,
    story_segment_method: str,
    recall_segment_method: str,
    sub_ids: list[str],
    rater_initials: str,
):
    # load story and recall segments
    story_segments, story_segment_method = load_story_segments(
        story_name=story_name, method=story_segment_method
    )
    recall_segments_list, recall_segment_method = load_recall_segments(
        story_name=story_name, method=recall_segment_method, sub_ids=sub_ids
    )

    # select correct subject
    for sub_id, recall_segments in recall_segments_list:
        rows = [
            {
                "recall_segment": r,
                "story_segment": s,
                "quality_score": "",
                "summary": "",
            }
            for r in recall_segments
            for s in story_segments
        ]

        df = pd.DataFrame(rows)

        output_dir = (
            Path("data")
            / "stories-and-recalls"
            / story_name
            / "recall_quality"
            / "human"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        param_str = (
            f"{sub_id}-{rater_initials}"
            f"-ssm_{story_segment_method}-rsm_{recall_segment_method}"
        )
        output_path = output_dir / f"{param_str}.csv"
        df.to_csv(output_path, index=False)
        print(f"Generated {output_path}")


if __name__ == "__main__":
    story_name = "pieman"
    story_segment_method = "sentences_corrected"
    recall_segment_method = "sentences"
    sub_ids = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006"]
    rater_initials = "GKP"
    generate_file(
        story_name=story_name,
        story_segment_method=story_segment_method,
        recall_segment_method=recall_segment_method,
        sub_ids=sub_ids,
        rater_initials=rater_initials,
    )
