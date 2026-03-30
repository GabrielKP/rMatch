from itertools import product
from pathlib import Path

import pandas as pd

from rmatch.load import load_story_segments


def prepare_human_causality_file(
    story_name: str,
    story_segmentation_method: str,
    rater_initials: str,
) -> None:
    story_segments, story_segmentation_method = load_story_segments(
        story_name=story_name, method=story_segmentation_method
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
    param_str = f"{rater_initials}-ssm_{story_segmentation_method}"
    output_path = output_dir / f"{param_str}.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    story_name = "pieman"
    story_segmentation_method = "sentences_corrected"
    rater_initials = "GKP"
    prepare_human_causality_file(story_name, story_segmentation_method, rater_initials)
