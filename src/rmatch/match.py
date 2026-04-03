import argparse
import json
from pathlib import Path
from typing import Literal

from rmatch import console
from rmatch.matchers import initialize_matcher
from rmatch.utils import get_param_str


def remove_empty_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def load_story_segments(story_path: Path) -> tuple[list[str], str]:
    story_path = Path(story_path)
    if not story_path.is_file():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    if story_path.suffix.lower() == ".json":
        data = json.loads(story_path.read_text(encoding="utf-8"))
        if "segments" not in data:
            raise ValueError(
                f"Story JSON {story_path} must contain a 'segments' array (see README)."
            )
        raw = data["segments"]
        if not isinstance(raw, list):
            raise ValueError(f"Story JSON 'segments' must be a list in {story_path}")
        segments = [s for s in raw if isinstance(s, str) and s.strip()]
        if not segments:
            raise ValueError(f"No segments in {story_path}")
        method = data.get("segmentation_method")
        if not method or not isinstance(method, str):
            method = "json"
        return segments, method

    segments = remove_empty_lines(story_path.read_text(encoding="utf-8"))
    if not segments:
        raise ValueError(f"Story file is empty or has no  lines: {story_path}")
    return segments, "lines"


def _filter_subs(
    pairs: list[tuple[str, list[str]]],
    sub_ids: list[str] | None,
) -> list[tuple[str, list[str]]]:
    if sub_ids is None:
        return pairs
    want = set(sub_ids)
    out = [(sid, segs) for sid, segs in pairs if sid in want]
    if not out:
        raise ValueError(
            f"No matching subjects after --sub-ids filter (wanted {sub_ids!r})."
        )
    return out


def _load_recalls_json_obj(
    data: dict,
    source: str,
) -> tuple[list[tuple[str, list[str]]], str]:
    if "recalls" not in data:
        raise ValueError(f"Recalls JSON {source} must contain a 'recalls' object.")
    recalls = data["recalls"]
    if not isinstance(recalls, dict):
        raise ValueError(f"'recalls' must be an object in {source}")

    method = data.get("segmentation_method")
    if not method or not isinstance(method, str):
        method = "json"

    pairs: list[tuple[str, list[str]]] = []
    for sub_id in sorted(recalls.keys()):
        segs_raw = recalls[sub_id]
        if not isinstance(segs_raw, list):
            raise ValueError(f"recalls[{sub_id!r}] must be a list in {source}")
        segs = [s for s in segs_raw if isinstance(s, str) and s.strip()]
        if not segs:
            raise ValueError(f"No recall segments for subject {sub_id!r} in {source}")
        pairs.append((sub_id, segs))

    return pairs, method


def _load_recall_single_json(path: Path) -> tuple[list[tuple[str, list[str]]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _load_recalls_json_obj(data, str(path))


def _load_recall_single_txt(path: Path) -> tuple[list[tuple[str, list[str]]], str]:
    segs = remove_empty_lines(path.read_text(encoding="utf-8"))
    if not segs:
        raise ValueError(f"Recall file has no segments: {path}")
    return [(path.stem, segs)], "lines"


def _merge_recall_json_files(
    paths: list[Path],
) -> tuple[list[tuple[str, list[str]]], str]:
    merged: dict[str, list[str]] = {}
    method: str | None = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs, m = _load_recalls_json_obj(data, str(path))
        if method is None:
            method = m

        for sid, segs in pairs:
            if sid in merged:
                raise ValueError(
                    f"Duplicate subject id {sid!r} across recall JSON files "
                    "in directory."
                )
            merged[sid] = segs

    out_pairs = [(sid, merged[sid]) for sid in sorted(merged.keys())]
    return out_pairs, (method or "json")


def _load_recall_txt_dir(
    paths: list[Path],
) -> tuple[list[tuple[str, list[str]]], str]:
    pairs: list[tuple[str, list[str]]] = []
    for path in paths:
        segs = remove_empty_lines(path.read_text(encoding="utf-8"))
        if not segs:
            raise ValueError(f"Recall file has no segments: {path}")
        pairs.append((path.stem, segs))
    return pairs, "lines"


def load_recall_segments(recall_path: Path) -> tuple[list[tuple[str, list[str]]], str]:
    """Load recall segments per subject.

    Returns (list of (sub_id, segments), recall_method).
    """
    recall_path = Path(recall_path)

    # single file
    if recall_path.is_file():
        if recall_path.suffix.lower() == ".json":
            return _load_recall_single_json(recall_path)
        return _load_recall_single_txt(recall_path)

    # directory
    if not recall_path.is_dir():
        raise FileNotFoundError(f"Recall path not found: {recall_path}")

    json_files = sorted(recall_path.glob("*.json"))
    txt_files = sorted(recall_path.glob("*.txt"))
    if json_files and txt_files:
        raise ValueError(
            f"Recall directory {recall_path} contains both .json and .txt "
            "files; use one format."
        )
    if json_files:
        return _merge_recall_json_files(json_files)
    if txt_files:
        return _load_recall_txt_dir(txt_files)
    raise ValueError(f"Recall directory {recall_path} has no .json or .txt files.")


def build_story_recall_segments(
    story_path: Path,
    recall_path: Path,
) -> tuple[list[tuple[str, list[str], list[str]]], str, str]:
    """Build rows for Matcher.match when loading from paths (preloaded)."""
    story_segments, story_method = load_story_segments(story_path)
    recall_list, recall_method = load_recall_segments(recall_path)
    rows = [
        (sub_id, story_segments, recall_segments)
        for sub_id, recall_segments in recall_list
    ]
    return rows, story_method, recall_method


def match(
    story_segments: list[str],
    recall_segments: list[str],
    matcher: str = "anthropic",
    model_name: str | None = None,
    device: str | None = None,
    quantization: Literal["4bit", "8bit"] | None = None,
    batch_size: int = 4,
) -> list[tuple[int, list[int]]]:
    """Returns recall-to-story matches for a single subject.

    Parameters
    ----------
    story_segments: list[str]
        list of story segments
    recall_segments: list[str]
        list of recall segments
    matcher: str
        matcher to use

    Returns
    """
    matcher_obj = initialize_matcher(
        matcher_name=matcher,
        model_name=model_name,
        device=device,
        quantization=quantization,
        batch_size=batch_size,
    )
    story_recall_segments = [("default", story_segments, recall_segments)]
    output_dict = matcher_obj.match(
        story_name="in_memory_story",
        story_segmentation_method="lines",
        recall_segmentation_method="lines",
        story_recall_segments=story_recall_segments,
        output_scores=False,
    )
    return output_dict["ratings"]["default"]


def recall_output_dir(recall_path: Path) -> Path:
    """Directory for ratings JSON: parent of a file, or the recall directory."""
    recall_path = Path(recall_path)
    if recall_path.is_file():
        return recall_path.parent
    if recall_path.is_dir():
        return recall_path
    raise FileNotFoundError(f"Recall path not found: {recall_path}")


def run_matching(
    story_file: Path,
    recall_file: Path,
    matcher_name: str,
    story_name: str,
    model_name: str | None,
    device: str | None,
    quantization: Literal["4bit", "8bit"] | None,
    batch_size: int,
    track_emissions: bool,
) -> Path:
    story_file = Path(story_file)
    recall_file = Path(recall_file)

    (
        story_recall_segments,
        story_segmentation_method,
        recall_segmentation_method,
    ) = build_story_recall_segments(story_file, recall_file)

    matcher = initialize_matcher(
        matcher_name=matcher_name,
        model_name=model_name,
        device=device,
        quantization=quantization,
        batch_size=batch_size,
    )

    out_dir = recall_output_dir(recall_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    # track emissions if requested
    tracker = None
    if track_emissions:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name=f"rmatch-match-{matcher_name}",
            output_dir=str(out_dir),
        )
        tracker.start()

    try:
        output_dict = matcher.match(
            story_name=story_name,
            story_segmentation_method=story_segmentation_method,
            recall_segmentation_method=recall_segmentation_method,
            story_recall_segments=story_recall_segments,
        )
    finally:
        if tracker is not None:
            emissions_kg = tracker.stop()
            console.print(
                f"[green]Carbon emissions:[/green] {emissions_kg:.6f} kg CO2eq"
            )

    param_str = get_param_str(output_dict)
    output_path = out_dir / f"{param_str}.json"
    return matcher.save_to_json(output_dict, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match recall segments to story segments (txt or JSON)."
    )
    parser.add_argument(
        "story_file",
        type=Path,
        help="Story: line-separated .txt or .json format (see README).",
    )
    parser.add_argument(
        "recall_file",
        type=Path,
        help="Recall: .txt or .json file, or directory of .txt or .json files.",
    )
    parser.add_argument(
        "-M",
        "--matcher",
        choices=["anthropic", "reranker", "openai", "huggingface"],
        default="anthropic",
        help="Matcher to use. Default: anthropic.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default=None,
        help="Underlying model to use for matcher. If None, use Matcher's default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="[reranker, huggingface] Device for model (default: auto).",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="[huggingface] Quantization: '4bit' or '8bit'",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=4,
        help="[huggingface] Batch size.",
    )
    parser.add_argument(
        "--track-emissions",
        action="store_true",
        default=False,
        help="Track carbon emissions with CodeCarbon (output beside recall).",
    )

    args = parser.parse_args()
    story_name = str(args.story_file.stem)

    out = run_matching(
        story_file=args.story_file,
        recall_file=args.recall_file,
        matcher_name=args.matcher,
        story_name=story_name,
        model_name=args.model_name,
        device=args.device,
        quantization=args.quantization,
        batch_size=args.batch_size,
        track_emissions=args.track_emissions,
    )
    console.print(f"[green]Wrote[/green] {out}")


if __name__ == "__main__":
    main()
