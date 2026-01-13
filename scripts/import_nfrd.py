"""Imports data from the NFRD dataset."""

import argparse
import json
from pathlib import Path

import chardet
import numpy as np
from nltk.tokenize import sent_tokenize


def read_recall_transcript(recall_transcript_path: Path) -> str:
    encoding_dict = {
        "Windows-1252": "cp1252",
        "ascii": "ascii",
        "utf-8": "utf-8",
    }

    with open(recall_transcript_path, "rb") as f:
        raw_recall = f.read()
    encoding = chardet.detect(raw_recall)

    return recall_transcript_path.read_text(
        encoding=encoding_dict[encoding["encoding"]]  # type: ignore
    )


def import_nfrd_data(nfrd_dir: Path):
    dim_topic_model = 40
    window_size = 55
    step_size = 21

    if not nfrd_dir.exists():
        raise FileNotFoundError(f"NFRD directory {nfrd_dir} does not exist")

    # 2. import story data
    story_ids = ["pieman", "oregon", "eyespy", "baseball"]
    for story_id in story_ids:
        subfolder_name = (
            f"{story_id}_t{dim_topic_model}_v{window_size}_r{window_size}_s{step_size}"
        )

        # (A) load and save story segments
        story_text_path = Path(
            nfrd_dir,
            "story_transcript",
            f"{story_id}_transcript.txt",
        )
        story_text = Path(story_text_path).read_text()
        # clean text
        story_text = (
            story_text.replace("’", "'")
            .replace("“", '"')
            .replace("“", '"')
            .replace("\n", " ")
            .strip()
        )
        story_text_output = Path(
            "data", "stories-and-recalls", story_id, "transcripts", "text.txt"
        )
        if not story_text_output.parent.exists():
            story_text_output.parent.mkdir(parents=True)

        # original text
        story_text_output.write_text(story_text + "\n")

        # sentence segments
        story_sentence_segments = [sent for sent in sent_tokenize(story_text)]
        story_sentence_segments_output = Path(
            "data",
            "stories-and-recalls",
            story_id,
            "transcripts",
            "sentences.txt",
        )
        story_sentence_segments_output.write_text(
            "\n".join(story_sentence_segments) + "\n"
        )

        # (B) load recall segements
        story_id_recall = story_id if story_id != "oregon" else "oregontrail"
        recall_segments_paths = Path(
            nfrd_dir, "recall_transcript", f"recall_transcript_{story_id_recall}"
        ).glob("*.txt")
        for recall_segments_path in recall_segments_paths:
            sub_id = f"sub-{int(recall_segments_path.stem.split('_')[0][1:]):03d}"
            recall_text = read_recall_transcript(recall_segments_path)
            recall_text = (
                recall_text.replace("’", "'")
                .replace("“", '"')
                .replace("“", '"')
                .replace("\n", " ")
                .strip()
            )

            # original text
            recall_text_output = Path(
                "data",
                "stories-and-recalls",
                story_id,
                "recalls",
                "text",
                f"{sub_id}.txt",
            )
            recall_text_output.parent.mkdir(parents=True, exist_ok=True)
            recall_text_output.write_text(recall_text + "\n")

            # sentence segments
            recall_segments = [sent for sent in sent_tokenize(recall_text)]
            recall_segments_output = Path(
                "data",
                "stories-and-recalls",
                story_id,
                "recalls",
                "sentences",
                f"{sub_id}.txt",
            )
            recall_segments_output.parent.mkdir(parents=True, exist_ok=True)
            recall_segments_output.write_text("\n".join(recall_segments) + "\n")

        # (C) load recalled event identities by lda-hmm
        recalled_events_path = Path(
            nfrd_dir,
            "result_models",
            subfolder_name,
            "labels.npy",
        )
        recalled_events_list = np.load(recalled_events_path, allow_pickle=True)
        # load participant ids
        story_recall_models_ids_path = Path(
            nfrd_dir, "result_models", f"{subfolder_name}.npy"
        )
        _, _, p_ids = np.load(story_recall_models_ids_path, allow_pickle=True)

        recalled_events = dict()
        for p_id, recall in zip(p_ids, recalled_events_list):
            # skip None events -> recalls which were not matched to
            # any story event. Not sure why this happens.
            sub_id = f"sub-{int(Path(p_id).stem.split('_')[0][1:]):03d}"
            recalled_events[sub_id] = [
                int(event) for event in recall if not np.isnan(event)
            ]
        recalled_events = dict(sorted(recalled_events.items(), key=lambda x: x[0]))

        output_dict = {
            "story_name": story_id,
            "story_segment_method": "lda-hmm",
            "recall_segment_method": "lda-hmm",
            "model_name": "lda-hmm",
            "ratings": recalled_events,
        }

        # save as json
        recalled_events_output = Path(
            "data",
            "stories-and-recalls",
            story_id,
            "ratings",
            "lda_hmm.json",
        )
        recalled_events_output.parent.mkdir(parents=True, exist_ok=True)
        with open(recalled_events_output, "w") as f:
            f.write(json.dumps(output_dict) + "\n")

        print(f"Imported {story_id} lda-hmm segmentation & recalled events")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    if args.dir is None:
        nfrd_dir = Path("../Naturalistic-Free-Recall-Dataset")
    else:
        nfrd_dir = Path(args.dir)

    import_nfrd_data(nfrd_dir=nfrd_dir)
