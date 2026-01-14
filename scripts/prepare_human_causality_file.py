from itertools import product
from pathlib import Path

import pandas as pd

from recall_matrix.load import load_story_segments


def prepare_human_causality_file(
    story_name: str,
    story_segment_method: str,
    rater_initials: str,
) -> None:
    story_segments = load_story_segments(
        story_name=story_name, method=story_segment_method
    )

    rows = [
        {"cause_segment": effect, "effect_segment": cause, "causality_score": ""}
        for (cause, effect) in product(story_segments, story_segments)
        if cause != effect
    ]

    df = pd.DataFrame(rows)

    output_dir = (
        Path("data") / "stories-and-recalls" / story_name / "causality" / "human"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rater_initials}.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    story_name = "pieman"
    story_segment_method = "sentences_corrected"
    rater_initials = "GKP"
    prepare_human_causality_file(story_name, story_segment_method, rater_initials)
