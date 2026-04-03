from pathlib import Path

import pandas as pd

from rmatch.load import load_recall_segments, load_story_segments


def generate_file(
    story_name: str,
    story_segmentation_method: str,
    recall_segmentation_method: str,
    sub_ids: list[str],
    rater_initials: str,
):
    # load story and recall segments
    story_segments, story_segmentation_method = load_story_segments(
        story_name=story_name, method=story_segmentation_method
    )
    recall_segments_list, recall_segmentation_method = load_recall_segments(
        story_name=story_name, method=recall_segmentation_method, sub_ids=sub_ids
    )

    # select correct subject
    for sub_id, recall_segments in recall_segments_list:
        rows = [
            {
                "recall_segment": r,
                "story_segment": s,
                "quality_score": "0",
                "notes": "",
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
            f"-ssm_{story_segmentation_method}-rsm_{recall_segmentation_method}"
        )
        output_path = output_dir / f"{param_str}.csv"
        df.to_csv(output_path, index=False)
        print(f"Generated {output_path}")


if __name__ == "__main__":
    story_name = "pieman"
    story_segmentation_method = "sentences_corrected"
    recall_segmentation_method = "sentences"
    sub_ids = ["sub-001", "sub-003"]
    rater_initials = "Ishan"
    generate_file(
        story_name=story_name,
        story_segmentation_method=story_segmentation_method,
        recall_segmentation_method=recall_segmentation_method,
        sub_ids=sub_ids,
        rater_initials=rater_initials,
    )
