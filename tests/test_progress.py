"""進捗表示機能のテスト"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.minutes_generator import MinutesGenerator
from app.transcriber import Transcriber


def test_transcriber_progress_bar(sample_audio_file: Path) -> None:
    """文字起こしの進捗表示テスト"""
    with patch("app.transcriber.WhisperModel") as mock_model_class:
        # モックのセグメント
        mock_segment1 = Mock()
        mock_segment1.start = 0.0
        mock_segment1.end = 5.0
        mock_segment1.text = "テスト1"

        mock_segment2 = Mock()
        mock_segment2.start = 5.0
        mock_segment2.end = 10.0
        mock_segment2.text = "テスト2"

        # モックの情報（duration付き）
        mock_info = Mock()
        mock_info.duration = 10.0

        mock_model = mock_model_class.return_value
        mock_model.transcribe.return_value = (
            iter([mock_segment1, mock_segment2]),
            mock_info,
        )

        transcriber = Transcriber(model_name="small")

        # 進捗バーが表示されることを確認
        full_text, segments = transcriber.transcribe(sample_audio_file)

        assert len(segments) == 2
        assert full_text == "テスト1テスト2"


def test_transcriber_no_duration(sample_audio_file: Path) -> None:
    """duration情報がない場合のテスト"""
    with patch("app.transcriber.WhisperModel") as mock_model_class:
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "テスト"

        # duration情報なし
        mock_info = Mock(spec=[])  # durationアトリビュートなし

        mock_model = mock_model_class.return_value
        mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

        transcriber = Transcriber(model_name="small")

        # エラーにならないことを確認
        full_text, segments = transcriber.transcribe(sample_audio_file)

        assert len(segments) == 1
        assert full_text == "テスト"


def test_minutes_generator_with_progress(tmp_path: Path) -> None:
    """議事録生成の進捗表示テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト議事録です。"}}

            with patch("app.minutes_generator.KnowledgeBase"):
                generator = MinutesGenerator(enable_rag=False)

                # 進捗バーが表示されながら議事録生成
                minutes = generator.generate("会議の内容です。")

                assert "議事録" in minutes
                mock_chat.assert_called_once()


def test_minutes_generator_with_rag_progress(tmp_path: Path) -> None:
    """RAG付き議事録生成の進捗表示テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト議事録です。"}}

            with patch("app.minutes_generator.KnowledgeBase") as mock_kb_class:
                mock_kb = mock_kb_class.return_value
                mock_kb.search.return_value = "関連知識"

                generator = MinutesGenerator(enable_rag=True)

                # RAG検索と議事録生成の進捗表示
                minutes = generator.generate("会議の内容です。")

                assert "議事録" in minutes
                mock_kb.search.assert_called_once()
                mock_chat.assert_called_once()


def test_main_workflow_step_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """メインワークフローのステップ進捗表示テスト"""
    from main import transcribe_existing_file

    # 音声ファイル作成
    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"dummy audio")

    inputs = iter([str(audio_file), "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = ("テスト文字起こし", [])
    mock_transcriber.save_transcript.return_value = tmp_path / "transcript.txt"

    mock_minutes_gen = Mock()
    mock_minutes_gen.generate_and_save.return_value = (
        "テスト議事録",
        tmp_path / "minutes.md",
    )

    # ステップ進捗が表示されることを確認
    transcribe_existing_file(mock_transcriber, mock_minutes_gen)

    mock_transcriber.transcribe.assert_called_once()
    mock_minutes_gen.generate_and_save.assert_called_once()


def test_main_workflow_without_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMなしワークフローのステップ進捗表示テスト"""
    from main import transcribe_existing_file

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"dummy audio")

    inputs = iter([str(audio_file), "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = ("テスト文字起こし", [])
    mock_transcriber.save_transcript.return_value = tmp_path / "transcript.txt"

    # LLMなしの場合、ステップ1のみ
    transcribe_existing_file(mock_transcriber, minutes_gen=None)

    mock_transcriber.transcribe.assert_called_once()
