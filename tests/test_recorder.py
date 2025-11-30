"""app.recorderのテスト"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.recorder import AudioRecorder


def test_audio_recorder_init() -> None:
    """AudioRecorderの初期化テスト"""
    recorder = AudioRecorder()

    assert recorder.sample_rate == 16000
    assert recorder.channels == 1
    assert recorder.recording == []
    assert recorder.is_recording is False


def test_audio_recorder_init_with_params() -> None:
    """AudioRecorderのパラメータ指定初期化テスト"""
    recorder = AudioRecorder(sample_rate=48000, channels=2)

    assert recorder.sample_rate == 48000
    assert recorder.channels == 2


def test_list_devices(caplog: pytest.LogCaptureFixture) -> None:
    """list_devicesのテスト"""
    recorder = AudioRecorder()

    with patch("sounddevice.query_devices") as mock_query:
        mock_query.return_value = "Device list"
        recorder.list_devices()

    assert "利用可能なオーディオデバイス" in caplog.text
    assert "Device list" in caplog.text


def test_save(temp_audio_dir: Path, sample_audio_data: np.ndarray) -> None:
    """saveメソッドのテスト"""
    recorder = AudioRecorder()
    output_path = temp_audio_dir / "saved.wav"

    result_path = recorder.save(sample_audio_data, output_path)

    assert result_path.exists()
    assert result_path == output_path


def test_save_auto_path(
    temp_audio_dir: Path, sample_audio_data: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """saveメソッドの自動パス生成テスト"""
    # data/audio ディレクトリを一時ディレクトリに変更
    monkeypatch.setattr(
        "app.recorder.Path", lambda x: temp_audio_dir if x == "data/audio" else Path(x)
    )

    recorder = AudioRecorder()
    result_path = recorder.save(sample_audio_data)

    assert result_path.exists()
    assert "recording_" in result_path.name


def test_record_and_save_error() -> None:
    """record_and_saveのエラーハンドリングテスト"""
    recorder = AudioRecorder()

    # 空の録音データを返すようにモック
    with patch.object(recorder, "record", return_value=np.array([])):
        with pytest.raises(ValueError, match="録音データが空です"):
            recorder.record_and_save()


def test_record_with_duration() -> None:
    """recordメソッドの時間指定テスト"""
    recorder = AudioRecorder()

    # InputStreamのモック
    mock_stream = MagicMock()

    def setup_recording(*args: object, **kwargs: object) -> MagicMock:
        # InputStream内で録音データを設定
        recorder.recording = [np.ones((100, 1), dtype=np.float32)]
        recorder.is_recording = True
        return mock_stream

    with patch("sounddevice.InputStream", side_effect=setup_recording):
        with patch("sounddevice.sleep") as mock_sleep:
            result = recorder.record(duration=0.1)

            mock_sleep.assert_called_once_with(100)  # 0.1秒 = 100ミリ秒
            assert len(result) > 0


def test_record_keyboard_interrupt() -> None:
    """recordメソッドのKeyboardInterrupt処理テスト"""
    recorder = AudioRecorder()

    mock_stream = MagicMock()
    mock_stream.__enter__.side_effect = KeyboardInterrupt()

    with patch("sounddevice.InputStream", return_value=mock_stream):
        result = recorder.record(duration=None)

        assert len(result) == 0


def test_main_function() -> None:
    """mainのmain関数テスト（初期化のみ）"""
    import sys
    from unittest.mock import patch

    # argvを偽装してテストモードに
    with patch.object(sys, "argv", ["recorder.py"]):
        # メイン関数を実行せずインポートのみテスト
        from app import recorder

        assert recorder.AudioRecorder is not None
