from pathlib import Path

import pandas as pd
from nltk.tokenize import sent_tokenize


def generate_file(story_name: str, participant_id: str, rater_name: str) -> None:
    base = Path(__file__).resolve().parents[2]
    story_path = base / "data" / "nfrd" / "transcripts" / f"{story_name}.txt"
    recall_path = (
        base
        / "data"
        / "nfrd"
        / "recalls"
        / "text"
        / story_name
        / f"{participant_id}_{story_name}.txt"
    )

    with open(story_path, "r", encoding="utf-8") as f:
        story = f.read()

    with open(recall_path, "r", encoding="utf-8") as f:
        recall = f.read()

    story_segments = sent_tokenize(story)
    recall_segments = sent_tokenize(recall)

    rows = [
        {"recall_segment": r, "story_segment": s, "mi": ""}
        for r in recall_segments
        for s in story_segments
    ]

    df = pd.DataFrame(rows)

    output_dir = base / "data" / "nfrd" / "recalls" / "human_mi" / story_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{participant_id}_{story_name}_{rater_name}.csv"
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    generate_file("pieman", "P1", "dhruva")
