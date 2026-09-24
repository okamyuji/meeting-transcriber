"""app.ragのテスト"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ollama import EmbedResponse, ListResponse

from app.rag import KnowledgeBase, is_model_available


@pytest.fixture
def temp_knowledge_dir(tmp_path: Path) -> Path:
    """一時的なナレッジディレクトリ"""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    return knowledge_dir


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """一時的なキャッシュディレクトリ"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_knowledge_file(temp_knowledge_dir: Path) -> Path:
    """サンプルナレッジファイル"""
    md_file = temp_knowledge_dir / "test.md"
    content = """# テストプロジェクト

## 概要
これはテストプロジェクトです。

## 用語
- RAG: Retrieval-Augmented Generation
- LLM: Large Language Model
"""
    md_file.write_text(content, encoding="utf-8")
    return md_file


def test_is_model_available_exact_match() -> None:
    """完全一致でモデルが見つかるテスト"""
    assert is_model_available("qwen2.5:7b", ["qwen2.5:7b"]) is True


def test_is_model_available_latest_suffix() -> None:
    """タグ省略指定でも:latestとして存在すれば見つかるテスト"""
    assert is_model_available("mxbai-embed-large", ["mxbai-embed-large:latest"]) is True


def test_is_model_available_not_found() -> None:
    """モデルが存在しないテスト"""
    assert is_model_available("mxbai-embed-large", ["qwen2.5:7b"]) is False


def test_knowledge_base_init(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """KnowledgeBaseの初期化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        assert kb.knowledge_dir == temp_knowledge_dir
        assert kb.cache_dir == temp_cache_dir
        assert kb.embed_model == "mxbai-embed-large"


def test_knowledge_base_init_model_with_latest_tag(
    temp_knowledge_dir: Path, temp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """埋め込みモデルがtagなしでも:latestとして存在すれば警告が出ないテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large:latest"}])

        with caplog.at_level("WARNING"):
            KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        assert "見つかりません" not in caplog.text
        assert "接続できません" not in caplog.text


def test_knowledge_base_empty_dir(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """空のナレッジディレクトリのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        assert len(kb.knowledge_chunks) == 0
        assert len(kb.embeddings) == 0


def test_knowledge_base_load(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """ナレッジベースのロードテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            assert len(kb.knowledge_chunks) > 0
            assert all("source" in chunk for chunk in kb.knowledge_chunks)
            assert all("title" in chunk for chunk in kb.knowledge_chunks)
            assert all("content" in chunk for chunk in kb.knowledge_chunks)


def test_split_into_chunks(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """チャンク分割のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        content = """# セクション1
内容1

## サブセクション
内容2

# セクション2
内容3
"""
        chunks = kb._split_into_chunks(content, "test.md")

        assert len(chunks) >= 2
        assert chunks[0]["title"] == "セクション1"
        assert "内容1" in chunks[0]["content"]


def test_compute_file_hash(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """ファイルハッシュ計算のテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2]])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        hash1 = kb._compute_file_hash(sample_knowledge_file)
        hash2 = kb._compute_file_hash(sample_knowledge_file)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5ハッシュは32文字


def test_get_embedding(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """埋め込み取得のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]])

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            embedding = kb._get_embedding("テストテキスト")

            assert len(embedding) == 5
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_embed.assert_called_with(
                model="mxbai-embed-large", input="テストテキスト", truncate=True
            )


def test_get_embedding_error(
    temp_knowledge_dir: Path, temp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """埋め込み取得エラー時のテスト（512トークン超のような長文でHTTP 500になるケース含む）"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed, caplog.at_level("ERROR"):
            mock_embed.side_effect = Exception("the input length exceeds the context length")

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            embedding = kb._get_embedding("テスト")

            assert embedding == []
            assert "埋め込み生成エラー" in caplog.text


def test_cosine_similarity(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """コサイン類似度計算のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        # 同じベクトル
        vec1 = [1.0, 0.0, 0.0]
        similarity = kb._cosine_similarity(vec1, vec1)
        assert abs(similarity - 1.0) < 0.001

        # 直交ベクトル
        vec2 = [0.0, 1.0, 0.0]
        similarity = kb._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

        # 空ベクトル
        similarity = kb._cosine_similarity([], vec1)
        assert similarity == 0.0


def test_search(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """検索機能のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            # ロード時のチャンク埋め込み（サンプルファイルは3チャンクに分割される）
            chunk_embeddings = [
                EmbedResponse(embeddings=[[1.0, 0.0, 0.0]]),
                EmbedResponse(embeddings=[[0.0, 1.0, 0.0]]),
                EmbedResponse(embeddings=[[0.0, 0.0, 1.0]]),
            ]
            mock_embed.side_effect = [
                *chunk_embeddings,
                EmbedResponse(embeddings=[[1.0, 0.1, 0.0]]),  # クエリ（チャンク1に類似）
            ]

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
            assert len(kb.knowledge_chunks) == 3

            result = kb.search("テストクエリ", top_k=1, threshold=0.5)

            assert isinstance(result, str)
            # 類似度が高いチャンクが返されるはず


def test_search_long_query_uses_windowed_max_similarity(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """クエリが512トークン相当より長い場合でも、後半のウィンドウで一致するチャンクを拾うテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.side_effect = [
                EmbedResponse(embeddings=[[1.0, 0.0, 0.0]]),  # チャンク1
                EmbedResponse(embeddings=[[0.0, 1.0, 0.0]]),  # チャンク2
                EmbedResponse(embeddings=[[0.0, 0.0, 1.0]]),  # チャンク3
            ]

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
            assert len(kb.knowledge_chunks) == 3

            # クエリ埋め込み呼び出しをここで設定し直す（ロード時の呼び出し分を消費済み）
            # ウィンドウ1はどのチャンクとも似ておらず、ウィンドウ2だけがチャンク3[0,0,1]と一致する
            mock_embed.side_effect = None
            mock_embed.return_value = EmbedResponse(embeddings=[[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

            long_query = "a" * 600  # 300文字ウィンドウ2つに分割される
            result = kb.search(long_query, top_k=1, threshold=0.5)

            call_kwargs = mock_embed.call_args.kwargs
            assert call_kwargs["truncate"] is True
            assert len(call_kwargs["input"]) == 2

            # ウィンドウ2でのみ一致するチャンク3（用語）が返る
            assert "用語" in result


def test_search_embed_error_returns_empty_string(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """検索時にクエリの埋め込み生成が失敗したら空文字を返すテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.side_effect = [
                EmbedResponse(embeddings=[[1.0, 0.0, 0.0]]),
                EmbedResponse(embeddings=[[0.0, 1.0, 0.0]]),
                EmbedResponse(embeddings=[[0.0, 0.0, 1.0]]),
            ]

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            mock_embed.side_effect = Exception("the input length exceeds the context length")
            result = kb.search("テストクエリ")

            assert result == ""


def test_search_empty_knowledge(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """空のナレッジベースでの検索テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        result = kb.search("テスト")

        assert result == ""


def test_cache_persistence(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """キャッシュの永続性テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])

            # 1回目のロード
            kb1 = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
            chunks1_count = len(kb1.knowledge_chunks)

            # 2回目のロード（キャッシュから）
            kb2 = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
            chunks2_count = len(kb2.knowledge_chunks)

            assert chunks1_count == chunks2_count


def test_refresh_cache(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """キャッシュリフレッシュのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            cache_file = temp_cache_dir / "embeddings.json"
            assert cache_file.exists()

            kb.refresh_cache()

            # キャッシュが再作成されているはず
            assert cache_file.exists()


def test_load_knowledge_reembeds_when_cached_embeddings_are_empty(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """空埋め込み([])がキャッシュされている場合、ハッシュが一致しても再埋め込みされるテスト"""
    file_hash = hashlib.md5(sample_knowledge_file.read_text(encoding="utf-8").encode()).hexdigest()
    cache_file = temp_cache_dir / "embeddings.json"
    cache_file.write_text(
        json.dumps(
            {
                "test.md": {
                    "hash": file_hash,
                    "chunks": [
                        {
                            "source": "test.md",
                            "title": "テストプロジェクト",
                            "content": "# テストプロジェクト\n",
                            "embedding": [],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])

        with patch("ollama.embed") as mock_embed:
            mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            # 空だったキャッシュは使われず、実際に埋め込みが呼ばれて非空になっているはず
            mock_embed.assert_called()
            assert all(chunk.get("embedding") for chunk in kb.knowledge_chunks)

            saved_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            assert all(c["embedding"] for c in saved_cache["test.md"]["chunks"])


def _load_kb_with_three_chunks(
    knowledge_dir: Path, cache_dir: Path, mock_embed: object
) -> KnowledgeBase:
    """sample_knowledge_file（3チャンク）を直交ベクトルで埋め込んだKnowledgeBaseを作る"""
    mock_embed.side_effect = [  # type: ignore[attr-defined]
        EmbedResponse(embeddings=[[1.0, 0.0, 0.0]]),
        EmbedResponse(embeddings=[[0.0, 1.0, 0.0]]),
        EmbedResponse(embeddings=[[0.0, 0.0, 1.0]]),
    ]
    kb = KnowledgeBase(knowledge_dir=knowledge_dir, cache_dir=cache_dir)
    mock_embed.side_effect = None  # type: ignore[attr-defined]
    return kb


def test_check_embed_model_warns_when_embed_model_missing(
    temp_knowledge_dir: Path, temp_cache_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """埋め込みモデルが一覧に無い場合はpullを促す警告を出すテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "qwen2.5:7b"}])

        with caplog.at_level("INFO"):
            KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

    assert "埋め込みモデル 'mxbai-embed-large' が見つかりません" in caplog.text
    assert "接続できません" not in caplog.text


def test_load_knowledge_reuses_valid_cache_without_embedding(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """ハッシュが一致し埋め込みが揃ったキャッシュは再計算せずに使うテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])
        KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
        mock_embed.reset_mock()

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

    mock_embed.assert_not_called()
    assert len(kb.knowledge_chunks) == 3
    assert kb.embeddings == [[0.1, 0.2, 0.3]] * 3


def test_load_knowledge_reembeds_when_file_changed(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """ファイル内容が変わってハッシュが一致しない場合は再計算するテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])
        KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
        sample_knowledge_file.write_text("# 変更後\n新しい内容\n", encoding="utf-8")
        mock_embed.reset_mock()

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

    mock_embed.assert_called_once()
    assert [c["title"] for c in kb.knowledge_chunks] == ["変更後"]


def test_load_knowledge_does_not_cache_failed_embeddings(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """埋め込みに失敗したファイルはキャッシュに書き込まないテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        mock_embed.side_effect = Exception("connection refused")

        KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

    saved_cache = json.loads((temp_cache_dir / "embeddings.json").read_text(encoding="utf-8"))
    assert "test.md" not in saved_cache


def test_load_knowledge_drops_stale_empty_cache_when_embedding_fails_again(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """空埋め込みのキャッシュが残っていて再計算も失敗した場合、その項目を消すテスト"""
    file_hash = hashlib.md5(sample_knowledge_file.read_text(encoding="utf-8").encode()).hexdigest()
    cache_file = temp_cache_dir / "embeddings.json"
    cache_file.write_text(
        json.dumps({"test.md": {"hash": file_hash, "chunks": [{"content": "x", "embedding": []}]}}),
        encoding="utf-8",
    )

    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        mock_embed.side_effect = Exception("connection refused")

        KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

    assert "test.md" not in json.loads(cache_file.read_text(encoding="utf-8"))


def test_search_splits_query_into_fixed_windows(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """クエリを先頭から300文字ずつの窓に分け、埋め込みモデルとtruncate指定で1回だけ送るテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        kb = _load_kb_with_three_chunks(temp_knowledge_dir, temp_cache_dir, mock_embed)
        mock_embed.reset_mock()
        mock_embed.return_value = EmbedResponse(embeddings=[[1.0, 0.0, 0.0]] * 3)

        kb.search("a" * 300 + "b" * 300 + "c" * 10)

    mock_embed.assert_called_once_with(
        model="mxbai-embed-large", input=["a" * 300, "b" * 300, "c" * 10], truncate=True
    )


def test_search_logs_error_when_query_embedding_fails(
    temp_knowledge_dir: Path,
    temp_cache_dir: Path,
    sample_knowledge_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """クエリの埋め込みに失敗したらエラーを記録して空文字を返すテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        kb = _load_kb_with_three_chunks(temp_knowledge_dir, temp_cache_dir, mock_embed)
        mock_embed.side_effect = Exception("boom")

        with caplog.at_level("ERROR"):
            result = kb.search("テストクエリ")

    assert result == ""
    assert "埋め込み生成エラー: boom" in caplog.text


def test_search_returns_empty_when_query_embeddings_empty(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """埋め込みAPIが空の結果を返したら空文字を返すテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        kb = _load_kb_with_three_chunks(temp_knowledge_dir, temp_cache_dir, mock_embed)
        mock_embed.return_value = EmbedResponse(embeddings=[])

        assert kb.search("テストクエリ") == ""


def test_search_returns_empty_when_no_chunk_reaches_threshold(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """どのチャンクも閾値に届かなければ空文字を返すテスト"""
    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        kb = _load_kb_with_three_chunks(temp_knowledge_dir, temp_cache_dir, mock_embed)
        mock_embed.return_value = EmbedResponse(embeddings=[[-1.0, -1.0, -1.0]])

        assert kb.search("テストクエリ", threshold=0.5) == ""


def test_split_into_chunks_caps_long_line_to_chunk_size(
    temp_knowledge_dir: Path, temp_cache_dir: Path
) -> None:
    """改行のない長い段落も、埋め込みモデルに収まるchunk_size以下に分割するテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)
    paragraph = "あ" * 1200

    chunks = kb._split_into_chunks(f"# 長文\n{paragraph}\n", "long.md", chunk_size=500)

    assert all(len(c["content"]) <= 500 for c in chunks)
    assert "".join(c["content"] for c in chunks) == f"# 長文\n{paragraph}"
    assert {c["title"] for c in chunks} == {"長文"}


def test_load_knowledge_reembeds_cache_from_older_chunk_format(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """チャンク形式の版が異なるキャッシュは、ハッシュが一致しても再計算するテスト"""
    file_hash = hashlib.md5(sample_knowledge_file.read_text(encoding="utf-8").encode()).hexdigest()
    cache_file = temp_cache_dir / "embeddings.json"
    cache_file.write_text(
        json.dumps(
            {"test.md": {"hash": file_hash, "chunks": [{"content": "x", "embedding": [1.0]}]}}
        ),
        encoding="utf-8",
    )

    with patch("ollama.list") as mock_list, patch("ollama.embed") as mock_embed:
        mock_list.return_value = ListResponse(models=[{"model": "mxbai-embed-large"}])
        mock_embed.return_value = EmbedResponse(embeddings=[[0.1, 0.2, 0.3]])

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

    assert len(kb.knowledge_chunks) == 3
    saved = json.loads(cache_file.read_text(encoding="utf-8"))["test.md"]
    assert saved["version"] == KnowledgeBase.CACHE_VERSION
