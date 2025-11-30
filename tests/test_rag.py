"""app.ragのテスト"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.rag import KnowledgeBase


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


def test_knowledge_base_init(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """KnowledgeBaseの初期化テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        assert kb.knowledge_dir == temp_knowledge_dir
        assert kb.cache_dir == temp_cache_dir
        assert kb.embed_model == "mxbai-embed-large"


def test_knowledge_base_empty_dir(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """空のナレッジディレクトリのテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        assert len(kb.knowledge_chunks) == 0
        assert len(kb.embeddings) == 0


def test_knowledge_base_load(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """ナレッジベースのロードテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        with patch("ollama.embeddings") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            assert len(kb.knowledge_chunks) > 0
            assert all("source" in chunk for chunk in kb.knowledge_chunks)
            assert all("title" in chunk for chunk in kb.knowledge_chunks)
            assert all("content" in chunk for chunk in kb.knowledge_chunks)


def test_split_into_chunks(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """チャンク分割のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

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
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        hash1 = kb._compute_file_hash(sample_knowledge_file)
        hash2 = kb._compute_file_hash(sample_knowledge_file)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5ハッシュは32文字


def test_get_embedding(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """埋め込み取得のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        with patch("ollama.embeddings") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            embedding = kb._get_embedding("テストテキスト")

            assert len(embedding) == 5
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_get_embedding_error(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """埋め込み取得エラー時のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        with patch("ollama.embeddings") as mock_embed:
            mock_embed.side_effect = Exception("Connection error")

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            embedding = kb._get_embedding("テスト")

            assert embedding == []


def test_cosine_similarity(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """コサイン類似度計算のテスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

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
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        with patch("ollama.embeddings") as mock_embed:
            # チャンクの埋め込みとクエリの埋め込みを設定
            mock_embed.side_effect = [
                {"embedding": [1.0, 0.0, 0.0]},  # チャンク1
                {"embedding": [0.0, 1.0, 0.0]},  # チャンク2
                {"embedding": [1.0, 0.1, 0.0]},  # クエリ（チャンク1に類似）
            ]

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            result = kb.search("テストクエリ", top_k=1, threshold=0.5)

            assert isinstance(result, str)
            # 類似度が高いチャンクが返されるはず


def test_search_empty_knowledge(temp_knowledge_dir: Path, temp_cache_dir: Path) -> None:
    """空のナレッジベースでの検索テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

        result = kb.search("テスト")

        assert result == ""


def test_cache_persistence(
    temp_knowledge_dir: Path, temp_cache_dir: Path, sample_knowledge_file: Path
) -> None:
    """キャッシュの永続性テスト"""
    with patch("ollama.list") as mock_list:
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        with patch("ollama.embeddings") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}

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
        mock_list.return_value = {"models": [{"name": "mxbai-embed-large"}]}

        with patch("ollama.embeddings") as mock_embed:
            mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}

            kb = KnowledgeBase(knowledge_dir=temp_knowledge_dir, cache_dir=temp_cache_dir)

            cache_file = temp_cache_dir / "embeddings.json"
            assert cache_file.exists()

            kb.refresh_cache()

            # キャッシュが再作成されているはず
            assert cache_file.exists()
