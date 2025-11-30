"""app.transcriberのテスト"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.transcriber import Segment, Transcriber


def test_segment_init() -> None:
    """Segmentの初期化テスト"""
    seg = Segment(start=0.0, end=1.5, text="テスト")

    assert seg.start == 0.0
    assert seg.end == 1.5
    assert seg.text == "テスト"


def test_segment_repr() -> None:
    """Segmentの__repr__テスト"""
    seg = Segment(start=1.2, end=3.4, text="こんにちは")

    assert "[1.20s -> 3.40s] こんにちは" in repr(seg)


def test_segment_to_dict() -> None:
    """Segmentのto_dictテスト"""
    seg = Segment(start=0.0, end=1.0, text="テスト")
    result = seg.to_dict()

    assert result["start"] == 0.0
    assert result["end"] == 1.0
    assert result["text"] == "テスト"


def test_transcriber_init() -> None:
    """Transcriberの初期化テスト"""
    with patch("app.transcriber.WhisperModel") as mock_model_class:
        transcriber = Transcriber(model_name="small", device="cpu", compute_type="int8")

        assert transcriber.model_name == "small"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        mock_model_class.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_transcriber_japanese_model_mapping() -> None:
    """Transcriberの日本語モデルマッピングテスト"""
    with patch("app.transcriber.WhisperModel") as mock_model_class:
        Transcriber(model_name="large-v3-ja")

        # large-v3-jaはkotoba-tech/kotoba-whisper-v1.0にマッピングされる
        mock_model_class.assert_called_once_with(
            "kotoba-tech/kotoba-whisper-v1.0", device="cpu", compute_type="int8"
        )


def test_transcribe_file_not_found() -> None:
    """transcribeメソッドのファイル未検出テスト"""
    with patch("app.transcriber.WhisperModel"):
        transcriber = Transcriber()

        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(Path("nonexistent.wav"))


def test_transcribe_success(sample_audio_file: Path, mock_whisper_model: Mock) -> None:
    """transcribeメソッドの成功テスト"""
    with patch("app.transcriber.WhisperModel", return_value=mock_whisper_model):
        transcriber = Transcriber()
        full_text, segments = transcriber.transcribe(sample_audio_file)

        assert len(full_text) > 0
        assert len(segments) > 0
        assert isinstance(segments[0], Segment)


def test_save_transcript(temp_transcript_dir: Path) -> None:
    """save_transcriptメソッドのテスト"""
    with patch("app.transcriber.WhisperModel"):
        transcriber = Transcriber()

        full_text = "テスト文字起こし"
        segments = [
            Segment(start=0.0, end=1.0, text="テスト"),
            Segment(start=1.0, end=2.0, text="文字起こし"),
        ]

        output_path = temp_transcript_dir / "test_output.txt"
        result_path = transcriber.save_transcript(full_text, segments, output_path)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "テスト文字起こし" in content
        assert "00:00:00 -> 00:00:01" in content


def test_save_transcript_auto_path(
    temp_transcript_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_transcriptメソッドの自動パス生成テスト"""
    monkeypatch.setattr(
        "app.transcriber.Path",
        lambda x: temp_transcript_dir if x == "data/transcripts" else Path(x),
    )

    with patch("app.transcriber.WhisperModel"):
        transcriber = Transcriber()

        full_text = "テスト"
        segments: list = []

        result_path = transcriber.save_transcript(full_text, segments)

        assert result_path.exists()
        assert "transcript_" in result_path.name


def test_format_time() -> None:
    """_format_timeメソッドのテスト"""
    assert Transcriber._format_time(0) == "00:00:00"
    assert Transcriber._format_time(65) == "00:01:05"
    assert Transcriber._format_time(3665) == "01:01:05"
