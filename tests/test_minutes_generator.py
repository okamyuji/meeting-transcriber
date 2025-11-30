"""app.minutes_generatorのテスト"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.minutes_generator import MinutesGenerator


def test_minutes_generator_init() -> None:
    """MinutesGeneratorの初期化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        generator = MinutesGenerator(model="qwen2.5:7b")

        assert generator.model == "qwen2.5:7b"
        assert generator.base_url == "http://localhost:11434"


def test_minutes_generator_custom_model() -> None:
    """MinutesGeneratorのカスタムモデル指定テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "llama3.1:8b"}]}

        generator = MinutesGenerator(model="llama3.1:8b")

        assert generator.model == "llama3.1:8b"


def test_minutes_generator_model_not_found() -> None:
    """MinutesGeneratorのモデル未検出テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": []}

        # 警告は出るが、エラーにはならない
        generator = MinutesGenerator(model="qwen2.5:7b")
        assert generator.model == "qwen2.5:7b"


def test_minutes_generator_ollama_not_running() -> None:
    """MinutesGeneratorのOllama未起動テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.side_effect = Exception("Connection refused")

        # 警告は出るが、エラーにはならない
        generator = MinutesGenerator()
        assert generator.model == "qwen2.5:7b"


def test_generate_success() -> None:
    """generateメソッドの成功テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト議事録です。"}}

            generator = MinutesGenerator()
            transcript = "会議の内容です。"

            minutes = generator.generate(transcript)

            assert "議事録" in minutes
            mock_chat.assert_called_once()


def test_generate_with_title() -> None:
    """generateメソッドのタイトル付きテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator()
            minutes = generator.generate("会議の内容", meeting_title="週次ミーティング")

            assert "議事録" in minutes
            # 呼び出されたメッセージに会議タイトルが含まれているか確認
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[1]["content"]
            assert "週次ミーティング" in user_message


def test_generate_empty_response() -> None:
    """generateメソッドの空レスポンステスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": ""}}

            generator = MinutesGenerator()

            with pytest.raises(ValueError, match="議事録の生成に失敗しました"):
                generator.generate("テスト")


def test_generate_with_context() -> None:
    """generateメソッドの追加コンテキスト付きテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator()
            minutes = generator.generate("会議の内容", additional_context="参加者: 田中、佐藤")

            assert "議事録" in minutes
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[1]["content"]
            assert "田中、佐藤" in user_message


def test_save_minutes(temp_transcript_dir: Path) -> None:
    """save_minutesメソッドのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        generator = MinutesGenerator()

        minutes = "# 議事録\n\nテスト議事録"
        output_path = temp_transcript_dir / "test_minutes.md"

        result_path = generator.save_minutes(minutes, output_path)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "議事録" in content


def test_save_minutes_auto_path(temp_transcript_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_minutesメソッドの自動パス生成テスト"""
    monkeypatch.setattr(
        "app.minutes_generator.Path",
        lambda x: temp_transcript_dir if x == "data/transcripts" else Path(x),
    )

    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        generator = MinutesGenerator()

        minutes = "テスト議事録"
        result_path = generator.save_minutes(minutes)

        assert result_path.exists()
        assert "minutes_" in result_path.name


def test_generate_and_save(temp_transcript_dir: Path) -> None:
    """generate_and_saveメソッドのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator()

            transcript = "会議の内容"
            output_path = temp_transcript_dir / "output.md"

            minutes, saved_path = generator.generate_and_save(transcript, output_path=output_path)

            assert "議事録" in minutes
            assert saved_path.exists()


def test_minutes_generator_with_rag_enabled() -> None:
    """MinutesGeneratorのRAG有効化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("app.minutes_generator.KnowledgeBase") as mock_kb_class:
            generator = MinutesGenerator(enable_rag=True)
            mock_kb_class.assert_called_once()
            assert generator.knowledge_base is not None


def test_minutes_generator_with_rag_disabled() -> None:
    """MinutesGeneratorのRAG無効化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        generator = MinutesGenerator(enable_rag=False)
        assert generator.knowledge_base is None


def test_generate_with_rag() -> None:
    """generateメソッドのRAG統合テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト議事録です。"}}

            with patch("app.minutes_generator.KnowledgeBase") as mock_kb_class:
                mock_kb = mock_kb_class.return_value
                mock_kb.search.return_value = "**RAG情報**\nプロジェクトX: 新規開発プロジェクト"

                generator = MinutesGenerator(enable_rag=True)
                transcript = "プロジェクトXについて議論しました。"

                minutes = generator.generate(transcript)

                assert "議事録" in minutes
                mock_kb.search.assert_called_once_with(transcript, top_k=3)

                # システムプロンプトにRAG情報が含まれているか確認
                call_args = mock_chat.call_args
                messages = call_args.kwargs["messages"]
                system_message = messages[0]["content"]
                assert "参考情報" in system_message
