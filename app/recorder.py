"""音声録音モジュール"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf

from app.logger import setup_logger

logger = setup_logger(__name__)


class AudioRecorder:
    """音声録音クラス"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        """
        Args:
            sample_rate: サンプリングレート（Hz）。Whisperは16kHzが最適
            channels: チャンネル数（1=モノラル、2=ステレオ）
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording: list[npt.NDArray[np.float32]] = []
        self.is_recording = False

    def list_devices(self) -> None:
        """利用可能なオーディオデバイスを表示"""
        logger.info("\n=== 利用可能なオーディオデバイス ===")
        logger.info(sd.query_devices())
        logger.info("")

    def record(self, duration: float | None = None) -> npt.NDArray[np.float32]:
        """
        音声を録音（Enterキーで停止、または指定時間で自動停止）

        Args:
            duration: 録音時間（秒）。Noneの場合はEnterキーで停止

        Returns:
            録音された音声データ（numpy配列）
        """
        logger.info("\n🎙️  録音を開始します...")
        if duration:
            logger.info(f"📊 {duration}秒間録音します")
        else:
            logger.info("⏸️  Enterキーを押すと録音を停止します")

        self.recording = []
        self.is_recording = True

        def callback(
            indata: npt.NDArray[np.float32],
            _frames: int,
            _time: Any,
            status: sd.CallbackFlags,
        ) -> None:
            """録音コールバック関数"""
            if status:
                logger.warning(f"⚠️  警告: {status}")
            if self.is_recording:
                self.recording.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate, channels=self.channels, callback=callback
            ):
                if duration:
                    sd.sleep(int(duration * 1000))
                else:
                    input()  # Enterキー待ち
        except KeyboardInterrupt:
            logger.warning("\n⚠️  録音を中断しました")
        finally:
            self.is_recording = False

        if not self.recording:
            logger.error("❌ 録音データがありません")
            return np.array([])

        audio_data = np.concatenate(self.recording, axis=0)
        duration_sec = len(audio_data) / self.sample_rate
        logger.info(f"✅ 録音完了: {duration_sec:.1f}秒")

        return audio_data

    def save(self, audio_data: npt.NDArray[np.float32], output_path: Path | None = None) -> Path:
        """
        録音データをWAVファイルとして保存

        Args:
            audio_data: 音声データ
            output_path: 保存先パス。Noneの場合は自動生成

        Returns:
            保存されたファイルのパス
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("data/audio") / f"recording_{timestamp}.wav"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(output_path), audio_data, self.sample_rate)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"💾 保存完了: {output_path} ({file_size_mb:.2f} MB)")

        return output_path

    def record_and_save(
        self, duration: float | None = None, output_path: Path | None = None
    ) -> Path:
        """
        録音して保存（ワンステップ）

        Args:
            duration: 録音時間（秒）。Noneの場合はEnterキーで停止
            output_path: 保存先パス。Noneの場合は自動生成

        Returns:
            保存されたファイルのパス
        """
        audio_data = self.record(duration)
        if len(audio_data) == 0:
            raise ValueError("録音データが空です")
        return self.save(audio_data, output_path)


def main() -> None:
    """テスト用のメイン関数"""
    recorder = AudioRecorder()
    recorder.list_devices()

    logger.info("録音テストを開始します。")
    try:
        audio_path = recorder.record_and_save()
        logger.info(f"\n✨ テスト成功: {audio_path}")
    except Exception as e:
        logger.error(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    main()
