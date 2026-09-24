"""app.minutes_generatorのテスト"""

from pathlib import Path
from unittest.mock import patch

import pytest
from ollama import ListResponse

from app.minutes_generator import MinutesGenerator


def test_minutes_generator_init() -> None:
    """MinutesGeneratorの初期化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        generator = MinutesGenerator(model="qwen2.5:7b", enable_rag=False)

        assert generator.model == "qwen2.5:7b"
        assert generator.base_url == "http://localhost:11434"


def test_minutes_generator_init_does_not_warn_when_model_present(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ListResponseの実レスポンス形式でモデルが見つかり警告が出ないテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with caplog.at_level("WARNING"):
            MinutesGenerator(model="qwen2.5:7b", enable_rag=False)

        assert "見つかりません" not in caplog.text
        assert "接続できません" not in caplog.text


def test_minutes_generator_model_found_with_latest_tag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """タグなし指定でも:latestとして存在すれば警告が出ないテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "llama3.1:latest"}])

        with caplog.at_level("WARNING"):
            MinutesGenerator(model="llama3.1", enable_rag=False)

        assert "見つかりません" not in caplog.text


def test_minutes_generator_custom_model() -> None:
    """MinutesGeneratorのカスタムモデル指定テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "llama3.1:8b"}])

        generator = MinutesGenerator(model="llama3.1:8b", enable_rag=False)

        assert generator.model == "llama3.1:8b"


def test_minutes_generator_model_not_found() -> None:
    """MinutesGeneratorのモデル未検出テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[])

        # 警告は出るが、エラーにはならない
        generator = MinutesGenerator(model="qwen2.5:7b", enable_rag=False)
        assert generator.model == "qwen2.5:7b"


def test_minutes_generator_model_not_found_among_others(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """他のモデルは存在するが指定モデルが見つからない場合に警告するテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "llama3.1:8b"}])

        with caplog.at_level("INFO"):
            generator = MinutesGenerator(model="qwen2.5:7b", enable_rag=False)

        assert generator.model == "qwen2.5:7b"
        assert "見つかりません" in caplog.text
        assert "llama3.1:8b" in caplog.text


def test_minutes_generator_ollama_not_running() -> None:
    """MinutesGeneratorのOllama未起動テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.side_effect = Exception("Connection refused")

        # 警告は出るが、エラーにはならない
        generator = MinutesGenerator(enable_rag=False)
        assert generator.model == "qwen2.5:7b"


def test_generate_success() -> None:
    """generateメソッドの成功テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト議事録です。"}}

            generator = MinutesGenerator(enable_rag=False)
            transcript = "会議の内容です。"

            minutes = generator.generate(transcript)

            assert "議事録" in minutes
            mock_chat.assert_called_once()


def test_generate_with_title() -> None:
    """generateメソッドのタイトル付きテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator(enable_rag=False)
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
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": ""}}

            generator = MinutesGenerator(enable_rag=False)

            with pytest.raises(ValueError, match="議事録の生成に失敗しました"):
                generator.generate("テスト")


def test_generate_with_context() -> None:
    """generateメソッドの追加コンテキスト付きテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator(enable_rag=False)
            minutes = generator.generate("会議の内容", additional_context="参加者: 田中、佐藤")

            assert "議事録" in minutes
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[1]["content"]
            assert "田中、佐藤" in user_message


def test_save_minutes(temp_transcript_dir: Path) -> None:
    """save_minutesメソッドのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        generator = MinutesGenerator(enable_rag=False)

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
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        generator = MinutesGenerator(enable_rag=False)

        minutes = "テスト議事録"
        result_path = generator.save_minutes(minutes)

        assert result_path.exists()
        assert "minutes_" in result_path.name


def test_generate_and_save(temp_transcript_dir: Path) -> None:
    """generate_and_saveメソッドのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator(enable_rag=False)

            transcript = "会議の内容"
            output_path = temp_transcript_dir / "output.md"

            minutes, saved_path = generator.generate_and_save(transcript, output_path=output_path)

            assert "議事録" in minutes
            assert saved_path.exists()


def test_minutes_generator_with_rag_enabled() -> None:
    """MinutesGeneratorのRAG有効化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("app.minutes_generator.KnowledgeBase") as mock_kb_class:
            generator = MinutesGenerator(enable_rag=True)
            mock_kb_class.assert_called_once()
            assert generator.knowledge_base is not None


def test_minutes_generator_with_rag_disabled() -> None:
    """MinutesGeneratorのRAG無効化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        generator = MinutesGenerator(enable_rag=False)
        assert generator.knowledge_base is None


def test_generate_with_rag() -> None:
    """generateメソッドのRAG統合テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

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


def test_calculate_num_ctx_short_prompt_uses_minimum() -> None:
    """短いプロンプトはOllamaのデフォルト下限4096になるテスト"""
    assert MinutesGenerator._calculate_num_ctx(0) == 4096
    assert MinutesGenerator._calculate_num_ctx(100) == 4096


def test_calculate_num_ctx_rounds_up_to_1024_multiple() -> None:
    """プロンプト長+出力トークン数を1024単位に切り上げるテスト"""
    # 10000 + 2048(NUM_PREDICT) = 12048 -> 1024単位で切り上げ12288
    assert MinutesGenerator._calculate_num_ctx(10000) == 12288


def test_calculate_num_ctx_clamps_to_max_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """qwen2.5の最大コンテキスト32768を超える場合は上限に丸め、警告を出すテスト"""
    with caplog.at_level("WARNING"):
        num_ctx = MinutesGenerator._calculate_num_ctx(100000)

    assert num_ctx == 32768
    assert "切り詰め" in caplog.text or "truncat" in caplog.text.lower()


def test_generate_passes_num_ctx_sized_to_prompt() -> None:
    """generateがプロンプト長に応じたnum_ctxをOllamaに渡すテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with patch("ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "# 議事録\n\nテスト"}}

            generator = MinutesGenerator(enable_rag=False)
            transcript = "会議の内容です。" * 1500

            generator.generate(transcript)

            call_kwargs = mock_chat.call_args.kwargs
            messages = call_kwargs["messages"]
            expected_chars = len(messages[0]["content"]) + len(messages[1]["content"])
            expected_num_ctx = MinutesGenerator._calculate_num_ctx(expected_chars)

            assert expected_num_ctx > MinutesGenerator.MIN_NUM_CTX
            assert call_kwargs["model"] == "qwen2.5:7b"
            assert call_kwargs["options"] == {
                "temperature": 0.3,
                "num_predict": 2048,
                "num_ctx": expected_num_ctx,
            }


def test_calculate_num_ctx_does_not_warn_at_exact_max(caplog: pytest.LogCaptureFixture) -> None:
    """見積もりがちょうど上限32768なら警告せず上限を返すテスト"""
    with caplog.at_level("WARNING"):
        num_ctx = MinutesGenerator._calculate_num_ctx(32768 - 2048)

    assert num_ctx == 32768
    assert caplog.text == ""


def test_calculate_num_ctx_warns_one_over_max(caplog: pytest.LogCaptureFixture) -> None:
    """見積もりが上限を1トークンでも超えたら警告するテストで、推定値と上限を含める"""
    with caplog.at_level("WARNING"):
        num_ctx = MinutesGenerator._calculate_num_ctx(32768 - 2048 + 1)

    assert num_ctx == 32768
    assert "推定32769トークン" in caplog.text
    assert "上限32768トークン" in caplog.text


def test_calculate_num_ctx_keeps_exact_multiple_and_rounds_next() -> None:
    """1024の倍数ちょうどはそのまま、1超えると次の倍数に切り上げるテスト"""
    assert MinutesGenerator._calculate_num_ctx(8192 - 2048) == 8192
    assert MinutesGenerator._calculate_num_ctx(8192 - 2048 + 1) == 9216


def test_calculate_num_ctx_minimum_boundary() -> None:
    """下限4096の境界で、4097以上は切り上げた値になるテスト"""
    assert MinutesGenerator._calculate_num_ctx(4096 - 2048) == 4096
    assert MinutesGenerator._calculate_num_ctx(4096 - 2048 + 1) == 5120
