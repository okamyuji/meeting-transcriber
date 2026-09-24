"""LLMを使った議事録生成モジュール（Ollama - 完全ローカル実行）"""

from datetime import datetime
from pathlib import Path

import ollama
from tqdm import tqdm

from app.logger import setup_logger
from app.rag import KnowledgeBase, is_model_available

logger = setup_logger(__name__)


class MinutesGenerator:
    """議事録生成クラス（Ollama使用）"""

    NUM_PREDICT = 2048  # 出力トークン数の上限
    MIN_NUM_CTX = 4096  # Ollamaのデフォルトコンテキスト長
    MAX_NUM_CTX = 32768  # qwen2.5シリーズの最大コンテキスト長
    NUM_CTX_STEP = 1024  # num_ctxの丸め単位

    DEFAULT_SYSTEM_PROMPT = """あなたは優秀な日本語の議事録作成アシスタントです。
会議の文字起こしテキストから、読みやすく整理された議事録を作成してください。

以下の形式で出力してください：

# 議事録

## 📅 会議情報
- 日時: {datetime}
- 参加者: （推測される場合は記載）

## 📋 議題・トピック
会議で話し合われた主要なトピックを箇条書きで記載

## 💬 議論の要約
各トピックについて、どのような議論が行われたかを簡潔にまとめる

## ✅ 決定事項
会議で決まったことを箇条書きで記載

## 📝 アクションアイテム
- [ ] タスク内容（担当者、期限）

## 📌 その他・メモ
補足事項や次回の予定など

---

注意事項：
- 日本語の自然な文章で書いてください
- 話し言葉を書き言葉に整えてください
- 重要なポイントは太字や箇条書きで強調してください
- 不明確な部分は推測せず、「（不明）」と記載してください
"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        system_prompt: str | None = None,
        base_url: str = "http://localhost:11434",
        enable_rag: bool = True,
    ) -> None:
        """
        Args:
            model: 使用するOllamaモデル（qwen2.5:7b, llama3.1:8b, gemma2:9b等）
            system_prompt: カスタムシステムプロンプト
            base_url: OllamaサーバーのURL
            enable_rag: RAG（ナレッジベース参照）を有効にするか
        """
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.base_url = base_url

        # RAGの初期化
        self.knowledge_base: KnowledgeBase | None = None
        if enable_rag:
            try:
                self.knowledge_base = KnowledgeBase()
            except Exception as e:
                logger.warning(f"⚠️  RAG初期化に失敗: {e}")
                logger.info("   ナレッジベースなしで続行します")

        # Ollamaの接続確認
        try:
            available_models = [m["model"] for m in ollama.list()["models"]]

            if not available_models:
                logger.warning("⚠️  Ollamaモデルが見つかりません")
                logger.info("   以下のコマンドでモデルをダウンロードしてください：")
                logger.info(f"   ollama pull {model}")
            elif not is_model_available(model, available_models):
                logger.warning(f"⚠️  モデル '{model}' が見つかりません")
                logger.info(f"   利用可能なモデル: {', '.join(available_models)}")
                logger.info("   以下のコマンドでダウンロード：")
                logger.info(f"   ollama pull {model}")
            else:
                logger.info(f"🤖 LLM設定: model={model} (Ollama)")

        except Exception as e:
            logger.warning(f"⚠️  Ollamaサーバーに接続できません: {e}")
            logger.info("   Ollamaが起動しているか確認してください：")
            logger.info("   https://ollama.com/download")

    @classmethod
    def _calculate_num_ctx(cls, prompt_chars: int) -> int:
        """
        プロンプトの文字数からOllamaに渡すnum_ctxを見積もる

        Ollamaのデフォルトnum_ctx（4096）は、日本語の長い会議の文字起こしを
        黙って切り詰める（keep=4のシステムプロンプトと文字起こし冒頭だけが残る）。
        システムプロンプト＋ユーザープロンプトの文字数をトークン数の上限として使う。

        Args:
            prompt_chars: システムプロンプトとユーザープロンプトの合計文字数

        Returns:
            Ollamaに渡すnum_ctx（MIN_NUM_CTX〜MAX_NUM_CTXの範囲、1024単位）
        """
        # ponytail: 文字数をトークン数の上限とみなす簡易見積もり（qwen2.5の日本語は
        # 実測約0.7トークン/文字なので安全側）。MAX_NUM_CTXを超える会議は切り詰められる。
        # 対応が必要になったらチャンク分割要約へ切り替える。
        estimated_tokens = prompt_chars + cls.NUM_PREDICT

        if estimated_tokens > cls.MAX_NUM_CTX:
            logger.warning(
                f"⚠️  文字起こしが長いため、コンテキスト上限を超えて"
                f"切り詰められる可能性があります（推定{estimated_tokens}トークン "
                f"> 上限{cls.MAX_NUM_CTX}トークン）"
            )

        rounded = -(-estimated_tokens // cls.NUM_CTX_STEP) * cls.NUM_CTX_STEP
        return max(cls.MIN_NUM_CTX, min(rounded, cls.MAX_NUM_CTX))

    def generate(
        self,
        transcript: str,
        meeting_title: str | None = None,
        additional_context: str | None = None,
    ) -> str:
        """
        文字起こしテキストから議事録を生成

        Args:
            transcript: 文字起こしテキスト
            meeting_title: 会議タイトル（オプション）
            additional_context: 追加コンテキスト（参加者情報など）

        Returns:
            生成された議事録（Markdown形式）
        """
        logger.info("\n📝 議事録を生成中...")
        logger.info(f"📊 入力文字数: {len(transcript)}")

        # システムプロンプトに日時を埋め込み
        current_datetime = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        system_prompt = self.system_prompt.replace("{datetime}", current_datetime)

        # RAGで関連知識を検索
        with tqdm(
            total=3, desc="🔍 準備中", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        ) as pbar:
            pbar.set_description("🔍 RAG検索中")
            if self.knowledge_base:
                relevant_knowledge = self.knowledge_base.search(transcript, top_k=3)
                if relevant_knowledge:
                    system_prompt += "\n\n【参考情報】\n以下の情報を参考にしてください：\n\n"
                    system_prompt += relevant_knowledge
            pbar.update(1)

            # ユーザープロンプトを構築
            pbar.set_description("📋 プロンプト作成")
            user_prompt = "以下の会議の文字起こしから議事録を作成してください。\n\n"

            if meeting_title:
                user_prompt += f"【会議タイトル】\n{meeting_title}\n\n"

            if additional_context:
                user_prompt += f"【追加情報】\n{additional_context}\n\n"

            user_prompt += f"【文字起こし】\n{transcript}"
            pbar.update(1)

            pbar.set_description("🤖 LLM実行準備")
            pbar.update(1)

        num_ctx = self._calculate_num_ctx(len(system_prompt) + len(user_prompt))

        try:
            # LLM実行の進捗表示
            with tqdm(
                total=100,
                desc="🤖 議事録生成中",
                unit="%",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}%",
            ) as llm_pbar:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={
                        "temperature": 0.3,  # 創造性を抑えて正確性重視
                        "num_predict": self.NUM_PREDICT,  # 最大出力トークン数
                        "num_ctx": num_ctx,  # プロンプト長に応じたコンテキスト長
                    },
                )
                llm_pbar.update(100)

            minutes = str(response["message"]["content"])
            if not minutes:
                raise ValueError("議事録の生成に失敗しました（空のレスポンス）")

            logger.info("✅ 議事録生成完了")
            logger.info(f"📏 出力文字数: {len(minutes)}")

            return minutes

        except Exception as e:
            logger.error(f"❌ エラー: {e}")
            raise

    def save_minutes(
        self, minutes: str, output_path: Path | None = None, format: str = "markdown"
    ) -> Path:
        """
        議事録をファイルに保存

        Args:
            minutes: 議事録テキスト
            output_path: 保存先パス。Noneの場合は自動生成
            format: ファイル形式（"markdown" または "txt"）

        Returns:
            保存されたファイルのパス
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "md" if format == "markdown" else "txt"
            output_path = Path("data/transcripts") / f"minutes_{timestamp}.{ext}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(minutes)

        file_size_kb = output_path.stat().st_size / 1024
        logger.info(f"💾 議事録を保存: {output_path} ({file_size_kb:.2f} KB)")

        return output_path

    def generate_and_save(
        self,
        transcript: str,
        meeting_title: str | None = None,
        additional_context: str | None = None,
        output_path: Path | None = None,
    ) -> tuple[str, Path]:
        """
        議事録を生成して保存（ワンステップ）

        Args:
            transcript: 文字起こしテキスト
            meeting_title: 会議タイトル
            additional_context: 追加コンテキスト
            output_path: 保存先パス

        Returns:
            (議事録テキスト, 保存されたファイルのパス)
        """
        minutes = self.generate(transcript, meeting_title, additional_context)
        saved_path = self.save_minutes(minutes, output_path)
        return minutes, saved_path


def main() -> None:
    """テスト用のメイン関数"""
    import sys

    if len(sys.argv) < 2:
        logger.info("使い方: python -m app.minutes_generator <文字起こしファイルのパス>")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])

    if not transcript_path.exists():
        logger.error(f"❌ ファイルが見つかりません: {transcript_path}")
        sys.exit(1)

    from app.transcriber import extract_full_text

    # 文字起こしを読み込み
    with open(transcript_path, encoding="utf-8") as f:
        transcript = extract_full_text(f.read())

    # 議事録を生成
    generator = MinutesGenerator()
    minutes, saved_path = generator.generate_and_save(transcript)

    # プレビュー表示
    logger.info("\n" + "=" * 80)
    logger.info("【議事録プレビュー】")
    logger.info("=" * 80)
    logger.info(minutes[:1000] + ("..." if len(minutes) > 1000 else ""))


if __name__ == "__main__":
    main()
