"""RAG（Retrieval-Augmented Generation）モジュール - 完全ローカル実行"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import ollama

from app.logger import setup_logger

logger = setup_logger(__name__)


def is_model_available(name: str, available_models: list[str]) -> bool:
    """Ollamaのモデル名が利用可能か確認する

    `ollama pull mxbai-embed-large` のようにタグを省略してpullすると、
    `ollama list` はタグ付きの `mxbai-embed-large:latest` として返す。
    そのため単純な完全一致では見つからない。
    """
    return name in available_models or f"{name}:latest" in available_models


class KnowledgeBase:
    """Markdownベースのナレッジベース（RAG用）"""

    QUERY_WINDOW_SIZE = 300  # mxbai-embed-largeの512トークン制限に収まる文字数の窓

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        cache_dir: Path | None = None,
        embed_model: str = "mxbai-embed-large",
    ) -> None:
        """
        Args:
            knowledge_dir: ナレッジファイル（*.md）のディレクトリ
            cache_dir: 埋め込みキャッシュのディレクトリ
            embed_model: 埋め込みモデル名
        """
        self.knowledge_dir = knowledge_dir or Path("data/knowledge")
        self.cache_dir = cache_dir or Path(".rag_cache")
        self.embed_model = embed_model

        self.knowledge_chunks: list[dict[str, str]] = []
        self.embeddings: list[list[float]] = []

        # ディレクトリ作成
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 埋め込みモデルの確認
        self._check_embed_model()

        # ナレッジベースをロード
        self.load_knowledge()

    def _check_embed_model(self) -> None:
        """埋め込みモデルの存在確認"""
        try:
            available_models = [m["model"] for m in ollama.list()["models"]]

            if not is_model_available(self.embed_model, available_models):
                logger.warning(f"⚠️  埋め込みモデル '{self.embed_model}' が見つかりません")
                logger.info("   以下のコマンドでダウンロード：")
                logger.info(f"   ollama pull {self.embed_model}")
        except Exception as e:
            logger.warning(f"⚠️  Ollamaサーバーに接続できません: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """ファイルのハッシュ値を計算（キャッシュ検証用）"""
        content = file_path.read_text(encoding="utf-8")
        return hashlib.md5(content.encode()).hexdigest()

    def _split_into_chunks(
        self, content: str, file_name: str, chunk_size: int = 500
    ) -> list[dict[str, str]]:
        """
        テキストをチャンクに分割

        Args:
            content: Markdownコンテンツ
            file_name: ファイル名
            chunk_size: チャンクの文字数

        Returns:
            チャンクのリスト
        """
        chunks = []

        # セクションで分割（# で始まる行）
        lines = content.split("\n")
        current_section = ""
        current_title = file_name

        for line in lines:
            if line.startswith("#"):
                # 前のセクションを保存
                if current_section.strip():
                    chunks.append(
                        {
                            "source": file_name,
                            "title": current_title,
                            "content": current_section.strip(),
                        }
                    )
                current_title = line.strip("# ").strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"

                # チャンクサイズを超えたら分割
                if len(current_section) > chunk_size:
                    chunks.append(
                        {
                            "source": file_name,
                            "title": current_title,
                            "content": current_section.strip(),
                        }
                    )
                    current_section = ""

        # 最後のセクション
        if current_section.strip():
            chunks.append(
                {
                    "source": file_name,
                    "title": current_title,
                    "content": current_section.strip(),
                }
            )

        return chunks

    def load_knowledge(self) -> None:
        """ナレッジベースをロード"""
        md_files = list(self.knowledge_dir.glob("*.md"))

        if not md_files:
            logger.info(f"📚 ナレッジベースが空です: {self.knowledge_dir}")
            logger.info(
                "   data/knowledge/ に *.md ファイルを配置すると、議事録作成時に参照されます"
            )
            return

        logger.info(f"📚 ナレッジベースをロード中: {len(md_files)} ファイル")

        # キャッシュファイルのパス
        cache_file = self.cache_dir / "embeddings.json"
        cache_data = {}

        # 既存のキャッシュを読み込み
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cache_data = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️  キャッシュの読み込みに失敗: {e}")
                cache_data = {}

        # ナレッジファイルを処理
        all_chunks = []
        updated = False

        for md_file in md_files:
            file_hash = self._compute_file_hash(md_file)
            file_name = md_file.name

            # キャッシュが有効かチェック（埋め込みが空のチャンクを含むキャッシュは無効）
            cached_entry = cache_data.get(file_name)
            if (
                cached_entry is not None
                and cached_entry.get("hash") == file_hash
                and all(c.get("embedding") for c in cached_entry["chunks"])
            ):
                # キャッシュから復元
                cached_chunks = cached_entry["chunks"]
                all_chunks.extend(cached_chunks)
            else:
                # 新規、ファイルが更新された、または前回の埋め込みが失敗していた
                logger.info(f"   処理中: {file_name}")
                content = md_file.read_text(encoding="utf-8")
                chunks = self._split_into_chunks(content, file_name)

                # 埋め込みを生成
                for chunk in chunks:
                    embedding = self._get_embedding(chunk["content"])
                    chunk["embedding"] = embedding  # type: ignore[assignment]

                # 埋め込みが失敗したチャンクを含む場合はキャッシュに保存しない
                # （次回ロード時に再度埋め込みを試みるため）
                if all(c.get("embedding") for c in chunks):
                    cache_data[file_name] = {"hash": file_hash, "chunks": chunks}
                else:
                    cache_data.pop(file_name, None)
                all_chunks.extend(chunks)
                updated = True

        self.knowledge_chunks = all_chunks

        # 埋め込みを抽出
        self.embeddings = [chunk.get("embedding", []) for chunk in all_chunks]

        # キャッシュを保存
        if updated:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ ナレッジベース準備完了: {len(all_chunks)} チャンク")

    def _get_embedding(self, text: str) -> list[float]:
        """
        テキストの埋め込みベクトルを取得

        Args:
            text: 埋め込むテキスト

        Returns:
            埋め込みベクトル
        """
        try:
            # 旧 /api/embeddings (ollama.embeddings) はmxbai-embed-largeの512トークン制限を
            # 超える入力でHTTP 500を返す。truncate=Trueな新API (/api/embed) を使う。
            response = ollama.embed(model=self.embed_model, input=text, truncate=True)
            # 空のレスポンスはIndexErrorとして下のexceptで失敗扱いにする
            embedding: Any = response["embeddings"][0]
            return list(embedding)
        except Exception as e:
            logger.error(f"❌ 埋め込み生成エラー: {e}")
            return []

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        コサイン類似度を計算

        Args:
            vec1: ベクトル1
            vec2: ベクトル2

        Returns:
            コサイン類似度（0〜1）
        """
        if not vec1 or not vec2:
            return 0.0

        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> str:
        """
        クエリに関連する知識を検索

        Args:
            query: 検索クエリ
            top_k: 上位k件を返す
            threshold: 類似度の閾値（これ以下は除外）

        Returns:
            関連知識のテキスト（Markdown形式）
        """
        if not self.knowledge_chunks:
            return ""

        # クエリ全体を1リクエストで埋め込むと512トークン制限を超えて後半が切り捨てられるため、
        # 短い窓に分割し、各チャンクは窓ごとの最大類似度で評価する
        query_windows = [
            query[i : i + self.QUERY_WINDOW_SIZE]
            for i in range(0, len(query), self.QUERY_WINDOW_SIZE)
        ] or [query]

        try:
            response = ollama.embed(model=self.embed_model, input=query_windows, truncate=True)
            query_embeddings: list[list[float]] = list(response["embeddings"])
        except Exception as e:
            logger.error(f"❌ 埋め込み生成エラー: {e}")
            return ""

        if not query_embeddings:
            return ""

        # 各チャンクとの類似度を計算（窓ごとの最大値を採用）
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            if chunk_embedding:
                similarity = max(
                    self._cosine_similarity(qe, chunk_embedding) for qe in query_embeddings
                )
                if similarity >= threshold:
                    similarities.append((i, similarity))

        # 類似度でソート
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 上位k件を取得
        results = []
        for i, _similarity in similarities[:top_k]:
            chunk = self.knowledge_chunks[i]
            results.append(f"**{chunk['title']}** (出典: {chunk['source']})\n{chunk['content']}")

        if results:
            logger.info(f"🔍 RAG検索: {len(results)} 件の関連知識を発見")
            return "\n\n---\n\n".join(results)

        return ""

    def refresh_cache(self) -> None:
        """キャッシュをクリアして再構築"""
        cache_file = self.cache_dir / "embeddings.json"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("🔄 キャッシュをクリアしました")
        self.load_knowledge()
