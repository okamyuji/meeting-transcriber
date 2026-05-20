"""音声文字起こしモジュール（faster-whisper使用）"""

import time
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

from app.logger import setup_logger

logger = setup_logger(__name__)


class Segment:
    """文字起こしセグメント"""

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self) -> str:
        return f"[{self.start:.2f}s -> {self.end:.2f}s] {self.text}"

    def to_dict(self) -> dict[str, float | str]:
        return {"start": self.start, "end": self.end, "text": self.text}


class Transcriber:
    """音声文字起こしクラス"""

    # 日本語向け推奨モデル
    # faster-whisper は CTranslate2 形式のモデルを必要とします。
    # kotoba-tech/kotoba-whisper-v1.0 は HuggingFace Transformers 形式のため
    # 直接ロードできないので、CTranslate2 版の `-faster` サフィックス付きを指定します。
    JAPANESE_MODELS = {
        "small": "small",  # 軽量（CPU向け）
        "medium": "medium",  # バランス型
        "large-v3": "large-v3",  # 高精度（完走の安定性を優先する場合はこちら）
        "large-v3-ja": "kotoba-tech/kotoba-whisper-v1.0-faster",  # 日本語特化（CTranslate2版）
    }

    def __init__(self, model_name: str = "medium", device: str = "cpu", compute_type: str = "int8"):
        """
        Args:
            model_name: モデル名（small/medium/large-v3/large-v3-ja）
            device: "cpu" または "cuda"
            compute_type: 計算精度（"int8", "int16", "float16", "float32"）
                         CPUの場合は"int8"が推奨（メモリ節約＆高速）
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        # モデル名のマッピング
        actual_model = self.JAPANESE_MODELS.get(model_name, model_name)

        logger.info(
            f"🔄 モデルをロード中: {actual_model} (device={device}, compute_type={compute_type})"
        )
        start_time = time.time()

        self.model = WhisperModel(actual_model, device=device, compute_type=compute_type)

        load_time = time.time() - start_time
        logger.info(f"✅ モデルロード完了 ({load_time:.2f}秒)")

    def transcribe(
        self,
        audio_path: Path,
        language: str = "ja",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: dict | None = None,
    ) -> tuple[str, list[Segment]]:
        """
        音声ファイルを文字起こし

        Args:
            audio_path: 音声ファイルのパス
            language: 言語コード（"ja"=日本語）
            beam_size: ビームサーチのサイズ（大きいほど精度向上、処理時間増）
            vad_filter: 音声区間検出を使用するか（無音部分をスキップ）
            vad_parameters: VADのパラメータ（詳細設定）

        Returns:
            (全文テキスト, セグメントリスト)
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        logger.info(f"\n🎯 文字起こし開始: {audio_path.name}")
        logger.info(f"📊 設定: language={language}, beam_size={beam_size}, vad_filter={vad_filter}")

        start_time = time.time()

        # VADパラメータのデフォルト設定
        if vad_filter and vad_parameters is None:
            vad_parameters = {
                "threshold": 0.5,  # 音声判定の閾値
                "min_speech_duration_ms": 250,  # 最小音声長（ミリ秒）
                "min_silence_duration_ms": 2000,  # セグメント分割の無音長（ミリ秒）
            }

        # 文字起こし実行
        segments_generator, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
        )

        # セグメントを収集
        segments = []
        full_text_parts = []

        # 音声の長さを取得（進捗表示用）
        duration = info.duration if hasattr(info, "duration") else None

        # 進捗バーの設定
        pbar = tqdm(
            total=int(duration) if duration else None,
            desc="📝 文字起こし中",
            unit="秒",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            disable=duration is None,
        )

        try:
            for seg in segments_generator:
                segment = Segment(start=seg.start, end=seg.end, text=seg.text.strip())
                segments.append(segment)
                full_text_parts.append(segment.text)

                # 進捗を更新
                if duration:
                    pbar.update(int(seg.end) - pbar.n)
        finally:
            pbar.close()

        full_text = "".join(full_text_parts)

        elapsed_time = time.time() - start_time
        audio_duration = info.duration if hasattr(info, "duration") else 0

        logger.info(f"✅ 文字起こし完了 ({elapsed_time:.2f}秒)")
        logger.info(f"📝 セグメント数: {len(segments)}")
        logger.info(f"📏 文字数: {len(full_text)}")
        if audio_duration > 0:
            speed_ratio = audio_duration / elapsed_time
            logger.info(f"⚡ 処理速度: {speed_ratio:.2f}x（音声の{speed_ratio:.2f}倍速）")

        return full_text, segments

    def save_transcript(
        self,
        full_text: str,
        segments: list[Segment],
        output_path: Path | None = None,
        include_timestamps: bool = True,
    ) -> Path:
        """
        文字起こし結果をテキストファイルに保存

        Args:
            full_text: 全文テキスト
            segments: セグメントリスト
            output_path: 保存先パス。Noneの場合は自動生成
            include_timestamps: タイムスタンプを含めるか

        Returns:
            保存されたファイルのパス
        """
        if output_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("data/transcripts") / f"transcript_{timestamp}.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("文字起こし結果\n")
            f.write("=" * 80 + "\n\n")

            if include_timestamps and segments:
                f.write("【タイムスタンプ付き】\n\n")
                for seg in segments:
                    f.write(f"[{self._format_time(seg.start)} -> {self._format_time(seg.end)}]\n")
                    f.write(f"{seg.text}\n\n")

                f.write("\n" + "=" * 80 + "\n\n")

            f.write("【全文】\n\n")
            f.write(full_text + "\n")

        logger.info(f"💾 文字起こしを保存: {output_path}")
        return output_path

    @staticmethod
    def _format_time(seconds: float) -> str:
        """秒数を HH:MM:SS 形式に変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    """テスト用のメイン関数"""
    import sys

    if len(sys.argv) < 2:
        logger.info("使い方: python -m app.transcriber <音声ファイルのパス>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    # モデルサイズを選択（デフォルトはmedium）
    model_name = sys.argv[2] if len(sys.argv) > 2 else "medium"

    transcriber = Transcriber(model_name=model_name)
    full_text, segments = transcriber.transcribe(audio_path)

    # 結果を保存
    transcriber.save_transcript(full_text, segments)

    # プレビュー表示
    logger.info("\n" + "=" * 80)
    logger.info("【文字起こし結果プレビュー】")
    logger.info("=" * 80)
    logger.info(full_text[:500] + ("..." if len(full_text) > 500 else ""))


if __name__ == "__main__":
    main()
