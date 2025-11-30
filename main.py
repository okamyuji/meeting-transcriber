#!/usr/bin/env python3
"""会議文字起こし＆議事録作成アプリケーション - メインCLI"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from app.logger import setup_logger
from app.minutes_generator import MinutesGenerator
from app.recorder import AudioRecorder
from app.transcriber import Transcriber

logger = setup_logger(__name__)


def print_banner() -> None:
    """アプリケーションバナーを表示"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         🎙️  会議文字起こし＆議事録作成ツール  📝               ║
║                                                              ║
║   ローカル実行 / 日本語高精度 / faster-whisper + LLM          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    logger.info(banner)


def print_menu() -> str:
    """メインメニューを表示"""
    menu = """
【メニュー】
  1. 🎙️  録音 → 文字起こし → 議事録作成（フルワークフロー）
  2. 📁 既存の音声ファイルから文字起こし＆議事録作成
  3. 📝 既存の文字起こしから議事録作成
  4. 🎤 録音のみ（テスト用）
  5. 🔧 設定（モデル選択など）
  6. ❌ 終了

番号を選択してください: """
    return input(menu).strip()


def full_workflow(
    recorder: AudioRecorder,
    transcriber: Transcriber,
    minutes_gen: MinutesGenerator | None = None,
) -> None:
    """フルワークフロー: 録音 → 文字起こし → 議事録作成"""
    logger.info("\n" + "=" * 80)
    logger.info("🎬 フルワークフロー開始")
    logger.info("=" * 80)

    # 会議情報の入力
    meeting_title = input("\n📋 会議タイトル（省略可）: ").strip()
    additional_context = input("👥 参加者情報など（省略可）: ").strip()

    total_steps = 3 if minutes_gen else 2

    # ステップ1: 録音
    logger.info("\n" + "=" * 80)
    logger.info(f"📍 ステップ1/{total_steps}: 録音")
    logger.info("=" * 80)
    try:
        audio_path = recorder.record_and_save()
    except Exception as e:
        logger.error(f"❌ 録音エラー: {e}")
        return

    # ステップ2: 文字起こし
    logger.info("\n" + "=" * 80)
    logger.info(f"📍 ステップ2/{total_steps}: 文字起こし")
    logger.info("=" * 80)
    try:
        full_text, segments = transcriber.transcribe(audio_path)
        transcript_path = transcriber.save_transcript(full_text, segments)
    except Exception as e:
        logger.error(f"❌ 文字起こしエラー: {e}")
        return

    # ステップ3: 議事録作成（LLM利用可能な場合のみ）
    if minutes_gen:
        logger.info("\n" + "=" * 80)
        logger.info(f"📍 ステップ3/{total_steps}: 議事録作成")
        logger.info("=" * 80)
        try:
            minutes, minutes_path = minutes_gen.generate_and_save(
                full_text,
                meeting_title=meeting_title if meeting_title else None,
                additional_context=additional_context if additional_context else None,
            )

            logger.info("\n" + "=" * 80)
            logger.info("✨ すべての処理が完了しました！")
            logger.info("=" * 80)
            logger.info(f"📁 音声ファイル: {audio_path}")
            logger.info(f"📄 文字起こし: {transcript_path}")
            logger.info(f"📝 議事録: {minutes_path}")

        except Exception as e:
            logger.error(f"❌ 議事録生成エラー: {e}")
            logger.info(f"📄 文字起こしは保存されました: {transcript_path}")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("✨ 録音と文字起こしが完了しました！")
        logger.info("=" * 80)
        logger.info(f"📁 音声ファイル: {audio_path}")
        logger.info(f"📄 文字起こし: {transcript_path}")
        logger.info("\n💡 議事録を作成するには、Ollamaをインストールしてください。")


def transcribe_existing_file(
    transcriber: Transcriber, minutes_gen: MinutesGenerator | None = None
) -> None:
    """既存の音声ファイルから文字起こし＆議事録作成"""
    logger.info("\n" + "=" * 80)
    logger.info("📁 既存ファイルの処理")
    logger.info("=" * 80)

    audio_path_str = input("\n音声ファイルのパスを入力: ").strip()

    # 引用符を削除（シェルからのコピペ対応）
    audio_path_str = audio_path_str.strip('"').strip("'")

    audio_path = Path(audio_path_str).expanduser()

    if not audio_path.exists():
        logger.error(f"❌ ファイルが見つかりません: {audio_path}")
        logger.info(f"   入力されたパス: {audio_path_str}")
        logger.info(f"   解決後のパス: {audio_path.absolute()}")
        return

    # ファイル形式のチェック
    valid_extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    if audio_path.suffix.lower() not in valid_extensions:
        logger.error(f"❌ サポートされていないファイル形式です: {audio_path.suffix}")
        logger.info(f"   対応形式: {', '.join(valid_extensions)}")
        logger.info("   テキストファイルの場合は、メニュー3を選択してください。")
        return

    # 会議情報の入力
    meeting_title = input("📋 会議タイトル（省略可）: ").strip()
    additional_context = input("👥 参加者情報など（省略可）: ").strip()

    total_steps = 2 if minutes_gen else 1

    # ステップ1: 文字起こし
    logger.info("\n" + "=" * 80)
    logger.info(f"📍 ステップ1/{total_steps}: 文字起こし")
    logger.info("=" * 80)
    try:
        full_text, segments = transcriber.transcribe(audio_path)
        transcript_path = transcriber.save_transcript(full_text, segments)
        logger.info("✅ 文字起こし完了")
    except Exception as e:
        logger.error(f"❌ 文字起こしエラー: {e}")
        return

    # ステップ2: 議事録作成
    if minutes_gen:
        logger.info("\n" + "=" * 80)
        logger.info(f"📍 ステップ2/{total_steps}: 議事録作成")
        logger.info("=" * 80)
        try:
            minutes, minutes_path = minutes_gen.generate_and_save(
                full_text,
                meeting_title=meeting_title if meeting_title else None,
                additional_context=additional_context if additional_context else None,
            )

            logger.info("\n" + "=" * 80)
            logger.info("✨ すべての処理が完了しました！")
            logger.info("=" * 80)
            logger.info(f"📄 文字起こし: {transcript_path}")
            logger.info(f"📝 議事録: {minutes_path}")

        except Exception as e:
            logger.error(f"❌ 議事録生成エラー: {e}")
            logger.info(f"📄 文字起こしは保存されました: {transcript_path}")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("✨ 文字起こし完了！")
        logger.info("=" * 80)
        logger.info(f"📄 文字起こし: {transcript_path}")


def generate_minutes_from_transcript(minutes_gen: MinutesGenerator) -> None:
    """既存の文字起こしから議事録作成"""
    logger.info("\n" + "=" * 80)
    logger.info("📝 文字起こしから議事録作成")
    logger.info("=" * 80)

    transcript_path_str = input("\n文字起こしファイルのパスを入力: ").strip()

    # 引用符を削除（シェルからのコピペ対応）
    transcript_path_str = transcript_path_str.strip('"').strip("'")

    transcript_path = Path(transcript_path_str).expanduser()

    if not transcript_path.exists():
        logger.error(f"❌ ファイルが見つかりません: {transcript_path}")
        logger.info(f"   入力されたパス: {transcript_path_str}")
        logger.info(f"   解決後のパス: {transcript_path.absolute()}")
        return

    # ファイル形式のチェック
    valid_extensions = [".txt", ".md", ".text"]
    if transcript_path.suffix.lower() not in valid_extensions:
        logger.error(f"❌ サポートされていないファイル形式です: {transcript_path.suffix}")
        logger.info(f"   対応形式: {', '.join(valid_extensions)}")
        logger.info("   音声ファイルの場合は、メニュー2を選択してください。")
        return

    # 文字起こしを読み込み
    try:
        with open(transcript_path, encoding="utf-8") as f:
            transcript = f.read()
    except UnicodeDecodeError:
        logger.error("❌ ファイルの読み込みに失敗しました（UTF-8エンコーディングエラー）")
        logger.info("   ファイルがテキスト形式であることを確認してください。")
        return

    # 会議情報の入力
    meeting_title = input("📋 会議タイトル（省略可）: ").strip()
    additional_context = input("👥 参加者情報など（省略可）: ").strip()

    # 議事録作成
    try:
        minutes, minutes_path = minutes_gen.generate_and_save(
            transcript,
            meeting_title=meeting_title if meeting_title else None,
            additional_context=additional_context if additional_context else None,
        )

        logger.info("\n" + "=" * 80)
        logger.info("✨ 議事録作成完了！")
        logger.info("=" * 80)
        logger.info(f"📝 議事録: {minutes_path}")

    except Exception as e:
        logger.error(f"❌ 議事録生成エラー: {e}")


def recording_test(recorder: AudioRecorder) -> None:
    """録音のみテスト"""
    logger.info("\n" + "=" * 80)
    logger.info("🎤 録音テスト")
    logger.info("=" * 80)

    try:
        audio_path = recorder.record_and_save()
        logger.info(f"\n✨ 録音完了: {audio_path}")
    except Exception as e:
        logger.error(f"❌ 録音エラー: {e}")


def configure_settings() -> dict:
    """設定メニュー"""
    logger.info("\n" + "=" * 80)
    logger.info("🔧 設定")
    logger.info("=" * 80)

    logger.info("\n【Whisperモデルサイズ】")
    logger.info("  1. small   - 軽量・高速（メモリ: ~1GB）")
    logger.info("  2. medium  - バランス型（メモリ: ~2GB）⭐ 推奨")
    logger.info("  3. large-v3 - 高精度（メモリ: ~4GB）")

    model_choice = input("\nモデルを選択 [1-3] (デフォルト: 2): ").strip()
    model_map = {"1": "small", "2": "medium", "3": "large-v3", "": "medium"}
    model = model_map.get(model_choice, "medium")

    logger.info(f"\n✅ モデル設定: {model}")

    return {"model": model}


def main() -> None:
    """メインエントリーポイント"""
    # 環境変数の読み込み
    load_dotenv()

    print_banner()

    # 初期設定
    settings = {"model": "medium"}

    # コンポーネントの初期化
    logger.info("🔄 初期化中...")
    recorder = AudioRecorder()
    transcriber = None
    minutes_gen = None

    # LLMの初期化（Ollama）
    try:
        minutes_gen = MinutesGenerator()
    except Exception as e:
        logger.warning(f"⚠️  議事録生成機能が利用できません: {e}")
        logger.info("   録音と文字起こしは利用可能です。")
        logger.info("   議事録を作成するには、Ollamaをインストールしてください：")
        logger.info("   https://ollama.com/download\n")

    # メインループ
    while True:
        choice = print_menu()

        if choice == "1":
            # Transcriber の遅延初期化（モデルロードに時間がかかるため）
            if transcriber is None:
                transcriber = Transcriber(model_name=settings["model"])
            full_workflow(recorder, transcriber, minutes_gen)

        elif choice == "2":
            if transcriber is None:
                transcriber = Transcriber(model_name=settings["model"])
            transcribe_existing_file(transcriber, minutes_gen)

        elif choice == "3":
            if minutes_gen is None:
                logger.error("\n❌ 議事録生成機能が利用できません。")
                logger.info("   Ollamaをインストールして起動してください：")
                logger.info("   https://ollama.com/download")
            else:
                generate_minutes_from_transcript(minutes_gen)

        elif choice == "4":
            recording_test(recorder)

        elif choice == "5":
            settings = configure_settings()
            # モデル変更時は再初期化が必要
            transcriber = None

        elif choice == "6":
            logger.info("\n👋 終了します。お疲れさまでした！")
            sys.exit(0)

        else:
            logger.warning("\n❌ 無効な選択です。1-6の番号を入力してください。")

        input("\n⏎ Enterキーを押してメニューに戻る...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  中断されました。")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
