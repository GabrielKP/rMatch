from pathlib import Path

import pandas as pd

from recall_matrix.load import load_recall_segments, load_story_segments


def generate_file(
    story_name: str,
    story_segment_method: str,
    recall_segment_method: str,
    participant_id: str,
    rater_initials: str,
) -> None:
    # load story and recall segments
    story_segments = load_story_segments(
        story_name=story_name, method=story_segment_method
    )
    recall_segments_list = load_recall_segments(
        story_name=story_name, method=recall_segment_method, sub_ids=[participant_id]
    )

    # select correct subject
    for sub_id, recall_segments in recall_segments_list:
        if sub_id == participant_id:
            break
    else:
        raise ValueError(f"Participant {participant_id} not found")

    rows = [
        {"recall_segment": r, "story_segment": s, "quality": ""}
        for r in recall_segments
        for s in story_segments
    ]

    df = pd.DataFrame(rows)

    output_dir = (
        Path("data") / "stories-and-recalls" / story_name / "ratings" / "human_quality"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rater_initials}.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    story_name = "pieman"
    story_segment_method = "sentences_corrected"
    recall_segment_method = "sentences"
    participant_id = "sub-001"
    rater_initials = "GKP"
    generate_file(
        story_name,
        story_segment_method,
        recall_segment_method,
        participant_id,
        rater_initials,
    )
