"""Tests for file-loading helpers and match() in rmatch.matching."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rmatch.matching import load_recall_segments, load_story_segments, match
from tests.conftest import RECALL_SEGMENTS, STORY_SEGMENTS

# ── load_story_segments ───────────────────────────────────────────────────────


class TestLoadStorySegments:
    def test_txt_basic(self, story_txt):
        segments, method = load_story_segments(story_txt)
        assert segments == STORY_SEGMENTS
        assert method == "lines"

    def test_txt_ignores_blank_lines(self, tmp_path):
        p = tmp_path / "story.txt"
        p.write_text("\n".join(["seg1", "", "seg2", "", "seg3"]) + "\n")
        segments, _ = load_story_segments(p)
        assert segments == ["seg1", "seg2", "seg3"]

    def test_txt_whitespace_only_lines_ignored(self, tmp_path):
        p = tmp_path / "story.txt"
        p.write_text("seg1\n   \nseg2\n")
        segments, _ = load_story_segments(p)
        assert segments == ["seg1", "seg2"]

    def test_json_basic(self, story_json):
        segments, method = load_story_segments(story_json)
        assert segments == STORY_SEGMENTS
        assert method == "scenes"

    def test_json_default_method_when_key_missing(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(json.dumps({"segments": STORY_SEGMENTS}))
        segments, method = load_story_segments(p)
        assert segments == STORY_SEGMENTS
        assert method == "json"

    def test_json_empty_segmentation_method_defaults(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(
            json.dumps({"segments": STORY_SEGMENTS, "segmentation_method": ""})
        )
        _, method = load_story_segments(p)
        assert method == "json"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_story_segments(tmp_path / "nonexistent.txt")

    def test_json_missing_segments_key_raises(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(json.dumps({"other": "data"}))
        with pytest.raises(ValueError, match="segments"):
            load_story_segments(p)

    def test_json_segments_not_list_raises(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(json.dumps({"segments": "not a list"}))
        with pytest.raises(ValueError):
            load_story_segments(p)

    def test_txt_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(ValueError):
            load_story_segments(p)

    def test_json_empty_segments_array_raises(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(json.dumps({"segments": []}))
        with pytest.raises(ValueError):
            load_story_segments(p)

    def test_json_filters_non_string_segments(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(json.dumps({"segments": ["valid", None, 42, "also valid"]}))
        segments, _ = load_story_segments(p)
        assert segments == ["valid", "also valid"]

    def test_json_segmentation_method_non_string_defaults(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(
            json.dumps({"segments": STORY_SEGMENTS, "segmentation_method": 42})
        )
        _, method = load_story_segments(p)
        assert method == "json"

    def test_json_malformed_raises_json_decode_error(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text("{")
        with pytest.raises(json.JSONDecodeError):
            load_story_segments(p)

    def test_json_top_level_list_raises_value_error(self, tmp_path):
        p = tmp_path / "story.json"
        p.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="segments"):
            load_story_segments(p)


# ── load_recall_segments ──────────────────────────────────────────────────────


class TestLoadRecallSegments:
    def test_single_txt_basic(self, recall_txt):
        pairs, method = load_recall_segments(recall_txt)
        assert len(pairs) == 1
        sub_id, segs = pairs[0]
        assert sub_id == "recall"  # stem of recall.txt
        assert segs == RECALL_SEGMENTS
        assert method == "lines"

    def test_single_json_basic(self, recall_json):
        pairs, method = load_recall_segments(recall_json)
        assert len(pairs) == 1
        sub_id, segs = pairs[0]
        assert sub_id == "sub01"
        assert segs == RECALL_SEGMENTS
        assert method == "sub_sentences"

    def test_single_json_subjects_sorted(self, tmp_path):
        p = tmp_path / "recalls.json"
        p.write_text(
            json.dumps(
                {
                    "recalls": {"z_sub": ["seg1"], "a_sub": ["seg2"]},
                    "segmentation_method": "x",
                }
            )
        )
        pairs, _ = load_recall_segments(p)
        assert [pair[0] for pair in pairs] == ["a_sub", "z_sub"]

    def test_directory_of_txt_files(self, recall_dir_txt):
        pairs, method = load_recall_segments(recall_dir_txt)
        assert len(pairs) == 2
        assert method == "lines"
        subject_ids = {p[0] for p in pairs}
        assert subject_ids == {"sub01", "sub02"}

    def test_directory_of_json_files(self, recall_dir_json):
        pairs, method = load_recall_segments(recall_dir_json)
        assert len(pairs) == 2
        subject_ids = {p[0] for p in pairs}
        assert subject_ids == {"sub01", "sub02"}

    def test_directory_mixed_formats_raises(self, tmp_path):
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a.txt").write_text("seg\n")
        (d / "b.json").write_text(
            json.dumps({"recalls": {"s": ["seg"]}, "segmentation_method": "x"})
        )
        with pytest.raises(ValueError, match="both"):
            load_recall_segments(d)

    def test_directory_empty_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError):
            load_recall_segments(d)

    def test_directory_duplicate_subject_across_json_files_raises(self, tmp_path):
        d = tmp_path / "dupes"
        d.mkdir()
        (d / "a.json").write_text(
            json.dumps({"recalls": {"sub01": ["seg1"]}, "segmentation_method": "x"})
        )
        (d / "b.json").write_text(
            json.dumps({"recalls": {"sub01": ["seg2"]}, "segmentation_method": "x"})
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_recall_segments(d)

    def test_path_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_recall_segments(Path("/nonexistent/path"))

    def test_single_txt_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(ValueError):
            load_recall_segments(p)

    def test_single_json_missing_recalls_key_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"segmentation_method": "x"}))
        with pytest.raises(ValueError, match="recalls"):
            load_recall_segments(p)

    def test_single_json_recalls_not_dict_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"recalls": ["not", "a", "dict"]}))
        with pytest.raises(ValueError):
            load_recall_segments(p)

    def test_single_json_empty_subject_segments_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"recalls": {"sub01": []}, "segmentation_method": "x"}))
        with pytest.raises(ValueError):
            load_recall_segments(p)

    def test_directory_json_default_method(self, tmp_path):
        d = tmp_path / "recalls"
        d.mkdir()
        (d / "a.json").write_text(json.dumps({"recalls": {"s1": ["seg1"]}}))
        pairs, method = load_recall_segments(d)
        # No segmentation_method in file → defaults to "json"
        assert method == "json"

    def test_directory_txt_one_empty_file_raises(self, tmp_path):
        d = tmp_path / "recalls"
        d.mkdir()
        (d / "sub01.txt").write_text(
            "Alice and the Queen drank tea.\n", encoding="utf-8"
        )
        (d / "sub02.txt").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no segments"):
            load_recall_segments(d)

    def test_directory_no_txt_or_json_raises(self, tmp_path):
        d = tmp_path / "recalls"
        d.mkdir()
        (d / "notes.csv").write_text("a,b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no .json or .txt"):
            load_recall_segments(d)

    def test_directory_json_method_mismatch_logs_warning(self, tmp_path, caplog):
        import logging

        d = tmp_path / "recalls"
        d.mkdir()
        (d / "a.json").write_text(
            json.dumps(
                {"recalls": {"sub01": ["seg1"]}, "segmentation_method": "method_a"}
            )
        )
        (d / "b.json").write_text(
            json.dumps(
                {"recalls": {"sub02": ["seg2"]}, "segmentation_method": "method_b"}
            )
        )
        with caplog.at_level(logging.WARNING):
            load_recall_segments(d)
        warnings = [
            r.message
            for r in caplog.records
            if "recall method mismatch" in r.message.lower()
        ]
        assert len(warnings) == 1


# ── match ─────────────────────────────────────────────────────────────────────


def _make_mock_matcher(match_return_value=None, matcher_name: str = "anthropic"):
    """Return a mock Matcher instance with sensible defaults."""
    mock = MagicMock()
    if match_return_value is None:
        match_return_value = [(0, [0]), (1, [1])]
    mock.match.return_value = match_return_value
    mock.matcher_name = matcher_name
    mock.model_name = None
    return mock


class TestMatch:
    def test_output_dict_has_required_keys(self, story_txt, recall_txt):
        result = match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        assert "matcher_name" in result
        assert "story_name" in result
        assert "story_segmentation" in result
        assert "recall_segmentation" in result
        assert "matches" in result

    def test_output_dict_correct_values(self, story_txt, recall_txt):
        result = match(
            matcher=_make_mock_matcher(matcher_name="openai"),
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        assert result["matcher_name"] == "openai"
        assert result["story_name"] == "story"  # stem of story.txt
        assert result["story_segmentation"] == "lines"
        assert result["recall_segmentation"] == "lines"

    def test_output_file_created(self, story_txt, recall_txt, tmp_path):
        match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        json_files = list(tmp_path.glob("*.json"))
        # At least one output JSON file created
        assert len(json_files) >= 1

    def test_raises_if_output_exists_and_no_overwrite(
        self, story_txt, recall_txt, tmp_path
    ):
        # Create the expected output file first
        out_file = tmp_path / "anthropic-lines-lines.json"
        out_file.write_text("{}")
        with pytest.raises(FileExistsError):
            match(
                matcher=_make_mock_matcher(),
                story_file=story_txt,
                recall_file=recall_txt,
                overwrite=False,
            )

    def test_overwrite_true_succeeds_when_file_exists(
        self, story_txt, recall_txt, tmp_path
    ):
        out_file = tmp_path / "anthropic-lines-lines.json"
        out_file.write_text("{}")
        result = match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        assert "matches" in result

    def test_story_name_override(self, story_txt, recall_txt):
        result = match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            story_name="custom_name",
            overwrite=True,
        )
        assert result["story_name"] == "custom_name"

    def test_segmentation_overrides(self, story_txt, recall_txt):
        result = match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            story_segmentation="scenes",
            recall_segmentation="sentences",
            overwrite=True,
        )
        assert result["story_segmentation"] == "scenes"
        assert result["recall_segmentation"] == "sentences"

    def test_matches_dict_contains_subject(self, story_txt, recall_txt):
        result = match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        # "recall.txt" → subject ID is "recall"
        assert "recall" in result["matches"]

    def test_checkpoint_cleaned_up_after_success(self, story_txt, recall_txt, tmp_path):
        match(
            matcher=_make_mock_matcher(),
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        checkpoint_files = list(tmp_path.glob("*.checkpoint.json"))
        assert len(checkpoint_files) == 0

    def test_story_file_not_found_raises(self, recall_txt):
        with pytest.raises(FileNotFoundError):
            match(
                matcher=_make_mock_matcher(),
                story_file=Path("/nonexistent/story.txt"),
                recall_file=recall_txt,
            )

    def test_recall_file_not_found_raises(self, story_txt):
        with pytest.raises(FileNotFoundError):
            match(
                matcher=_make_mock_matcher(),
                story_file=story_txt,
                recall_file=Path("/nonexistent/recall.txt"),
            )

    def test_model_name_included_when_matcher_has_it(self, story_txt, recall_txt):
        mock_matcher = _make_mock_matcher(matcher_name="openai")
        mock_matcher.model_name = "gpt-4.1"
        result = match(
            matcher=mock_matcher,
            story_file=story_txt,
            recall_file=recall_txt,
            overwrite=True,
        )
        assert result.get("model_name") == "gpt-4.1"

    def test_match_with_json_story_and_recall(self, story_json, recall_json, tmp_path):
        result = match(
            matcher=_make_mock_matcher(),
            story_file=story_json,
            recall_file=recall_json,
            overwrite=True,
        )
        assert result["story_segmentation"] == "scenes"
        assert result["recall_segmentation"] == "sub_sentences"

    def test_match_with_recall_directory(self, story_txt, recall_dir_txt, tmp_path):
        # Directory has 2 subjects → matches dict should have 2 entries
        mock_matcher = _make_mock_matcher()
        mock_matcher.match.return_value = [(0, [0])]
        result = match(
            matcher=mock_matcher,
            story_file=story_txt,
            recall_file=recall_dir_txt,
            overwrite=True,
        )
        assert len(result["matches"]) == 2
        assert "sub01" in result["matches"]
        assert "sub02" in result["matches"]

    def test_keyboard_interrupt_writes_checkpoint_and_reraises(
        self, story_txt, recall_dir_txt, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        mock_matcher = _make_mock_matcher()
        mock_matcher.match.side_effect = KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            match(
                matcher=mock_matcher,
                story_file=story_txt,
                recall_file=recall_dir_txt,
                overwrite=True,
            )

        checkpoint_files = list(recall_dir_txt.glob("*.checkpoint.json"))
        assert len(checkpoint_files) == 1
        data = json.loads(checkpoint_files[0].read_text())
        assert data["checkpoint"] is True
        assert data["progress"]["reason"] == "KeyboardInterrupt"

    def test_generic_exception_does_not_write_checkpoint(
        self, story_txt, recall_dir_txt, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        mock_matcher = _make_mock_matcher()
        mock_matcher.match.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            match(
                matcher=mock_matcher,
                story_file=story_txt,
                recall_file=recall_dir_txt,
                overwrite=True,
            )

        assert len(list(recall_dir_txt.glob("*.checkpoint.json"))) == 0
