import argparse
import os
from pathlib import Path

import pandas as pd

from rmatch import get_logger
from scripts.segmenter_vllm import SegmenterVLLM

log = get_logger(__name__)


def segs_to_df(segments: list[str]) -> pd.DataFrame:
    data = {
        "recall_segment": list(range(1, len(segments) + 1)),
        "story_segment": [""] * len(segments),
        "text": segments,
    }

    return pd.DataFrame(data)


def process_directory(
    input_dir: str,
    output_dir: str,
    model_name: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float = 0.90,
    use_batch: bool = True,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    output_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_path}")

    transcript_files = list(input_path.glob("*.txt"))

    if not transcript_files:
        log.warning("No transcripts found")
        return

    log.info(f"Found {len(transcript_files)} transcript files")

    segmenter = SegmenterVLLM(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    if use_batch:
        log.info("Using batch mode")
        transcripts = []
        for file_path in transcript_files:
            with open(file_path, "r", encoding="utf-8") as f:
                transcripts.append(f.read())

        labels = [f.name for f in transcript_files]
        all_segments = segmenter.segment_batch(transcripts, labels)

        for file_path, segments in zip(transcript_files, all_segments):
            output_file = output_path / f"{file_path.stem}.csv"
            df = segs_to_df(segments)
            df.to_csv(output_file, index=False, encoding="utf-8")
            log.info(f"Saved: {output_file} ({len(segments)} segments)")
    else:
        log.info("Using sequential mode")
        for i, file_path in enumerate(transcript_files, 1):
            log.info(f"Processing {i}/{len(transcript_files)}: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read()

            try:
                segments = segmenter.segment(transcript)
            except Exception as e:
                log.error(f"Failed to segment {file_path.name}: {e}")
                segments = []

            output_file = output_path / f"{file_path.stem}.csv"
            df = segs_to_df(segments)
            df.to_csv(output_file, index=False, encoding="utf-8")
            log.info(f"Saved: {output_file} ({len(segments)} segments)")

    log.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment transcripts w/ vLLM")

    parser.add_argument(
        "input_dir",
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-batch",
        action="store_true",
    )

    args = parser.parse_args()
    if args.output_dir is None:
        output_dir = os.path.join(args.input_dir, "segmented")
    else:
        output_dir = args.output_dir

    process_directory(
        input_dir=args.input_dir,
        output_dir=output_dir,
        model_name=args.model,
        use_batch=not args.no_batch,
    )
