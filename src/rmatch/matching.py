import argparse
import datetime as dt
import json
from pathlib import Path

from rmatch import console
from rmatch.matchers import MatcherBase, get_matcher
from rmatch.utils import (
    atomic_write_json,
    get_logger,
    get_param_str,
)

log = get_logger(__name__)


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
        raise ValueError(f"Story file is empty or has no lines: {story_path}")
    return segments, "lines"


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
    warned_mismatch: bool = False
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs, m = _load_recalls_json_obj(data, str(path))
        if method is None:
            method = m
        elif method != m:
            if not warned_mismatch:
                log.warning(
                    f"At least two json have a recall method mismatch: {method} != {m}"
                )
                warned_mismatch = True

        for sid, segs in pairs:
            if sid in merged:
                raise ValueError(
                    f"Duplicate subject id {sid!r} across recall JSON files "
                    "in directory."
                )
            merged[sid] = segs

    out_pairs = sorted(merged.items())
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

    Returns
    -------
    recall_segments_list: list[tuple[str, list[str]]]
        list of (sub_id, segments)
    recall_method: str
        recall method
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


def recall_output_dir(recall_path: Path) -> Path:
    """Directory for ratings JSON: parent of a file, or the recall directory."""
    recall_path = Path(recall_path)
    if recall_path.is_file():
        return recall_path.parent
    if recall_path.is_dir():
        return recall_path
    raise FileNotFoundError(f"Recall path not found: {recall_path}")


def match(
    matcher: MatcherBase,
    story_file: Path,
    recall_file: Path,
    story_name: str | None = None,
    story_segmentation: str | None = None,
    recall_segmentation: str | None = None,
    overwrite: bool = False,
    **kwargs,
) -> dict:
    story_file = Path(story_file)
    recall_file = Path(recall_file)

    # load data
    story_segments, candidate_story_segmentation = load_story_segments(story_file)
    candidate_story_name = story_file.stem
    subs_and_recall_segments_list, candidate_recall_segmentation = load_recall_segments(
        recall_file
    )

    # determine story name and segmentations (user-specified overrides auto-detected)
    story_name = story_name or candidate_story_name
    story_segmentation = story_segmentation or candidate_story_segmentation
    recall_segmentation = recall_segmentation or candidate_recall_segmentation

    output_dict: dict = {
        "matcher_name": matcher.matcher_name,
        "story_name": story_name,
        "story_segmentation": story_segmentation,
        "recall_segmentation": recall_segmentation,
    }

    # output dir
    param_str = get_param_str(output_dict)
    out_dir = recall_output_dir(recall_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{param_str}.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {out_path}")
    console.print(f"Output path: {out_path}")

    model_name = getattr(matcher, "model_name", None)
    if model_name is not None:
        output_dict["model_name"] = model_name

    checkpoint_path = out_dir / f"{param_str}.checkpoint.json"

    def _save_match_checkpoint(
        matches_dict_local: dict[str, list[tuple[int, list[int]]]],
        *,
        reason: str | None = None,
    ) -> None:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        checkpoint_data = {
            **output_dict,
            "matches": matches_dict_local,
            "checkpoint": True,
            "progress": {
                "subjects_total": len(subs_and_recall_segments_list),
                "subjects_done": len(matches_dict_local),
                "subject_ids_done": list(matches_dict_local.keys()),
                "timestamp": ts,
                **({"reason": reason} if reason else {}),
            },
        }
        atomic_write_json(checkpoint_path, checkpoint_data, indent=2)
        log.info(f"Saved checkpoint to {checkpoint_path}")

    matches_dict: dict[str, list[tuple[int, list[int]]]] = {}
    try:
        for sub_id, recall_segments in subs_and_recall_segments_list:
            matches = matcher.match(
                story_segments=story_segments, recall_segments=recall_segments
            )
            matches_dict[sub_id] = matches
            _save_match_checkpoint(matches_dict)
    except KeyboardInterrupt:
        _save_match_checkpoint(matches_dict, reason="KeyboardInterrupt")
        console.print(
            f"[yellow]Interrupted; checkpoint written to[/yellow] {checkpoint_path}"
        )
        raise

    # add matches to output dict
    output_dict["matches"] = matches_dict

    atomic_write_json(out_path, output_dict)
    log.info(f"Saved matches to {out_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info(f"Removed checkpoint {checkpoint_path}")

    return output_dict


def main() -> None:
    # main CLI entry point
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
        choices=["anthropic", "openai", "mac", "cuda", "huggingface"],
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
        "--window-size",
        type=int,
        default=None,
        help=(
            "[anthropic, openai, huggingface] "
            "Size of recall context window (+/-). Default: 5."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="[anthropic, openai] Estimate cost without calling the API.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="[huggingface] Device for model (default: auto).",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="[huggingface] Quantization: '4bit' or '8bit'.",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=None,
        help="[huggingface] Batch size. Default: 4.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="[huggingface, cuda] max_new_tokens for the matcher. Default: 64.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "[cuda] Maximum sequence length of the model (prompt + generation). "
            "Default: 90000. "
            "Set to 'auto' to auto-detect the max from model config."
            "Set this lower to reduce memory usage. "
        ),
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        default=None,
        help="[huggingface] Print verbose errors.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=[
            "primary",
            "primary_no_story",
            "primary_no_cot",
            "primary_no_story_no_cot",
            "secondary",
        ],
        default=None,
        help="[anthropic, openai, huggingface] Prompt type. Default is 'primary'.",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output file.",
    )

    args = parser.parse_args()

    # initialize matcher
    matcher_kwargs = {
        k: v
        for k, v in vars(args).items()
        if v is not None
        and k not in ["matcher", "story_file", "recall_file", "overwrite"]
    }
    matcher = get_matcher(args.matcher, **matcher_kwargs)

    match(
        matcher=matcher,
        story_file=args.story_file,
        recall_file=args.recall_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
