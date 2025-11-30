"""main.pyのテスト"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from main import (
    configure_settings,
    generate_minutes_from_transcript,
    print_banner,
    recording_test,
    transcribe_existing_file,
)


def test_print_banner(caplog: pytest.LogCaptureFixture) -> None:
    """print_bannerのテスト"""
    print_banner()

    assert "会議文字起こし＆議事録作成ツール" in caplog.text


def test_configure_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_settingsのテスト"""
    # ユーザー入力をモック
    inputs = iter(["2"])  # mediumを選択
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    result = configure_settings()

    assert result["model"] == "medium"


def test_configure_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_settingsのデフォルト選択テスト"""
    inputs = iter([""])  # デフォルト
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    result = configure_settings()

    assert result["model"] == "medium"


def test_recording_test() -> None:
    """recording_testのテスト"""
    mock_recorder = Mock()
    mock_recorder.record_and_save.return_value = Path("/tmp/test.wav")

    recording_test(mock_recorder)

    mock_recorder.record_and_save.assert_called_once()


def test_recording_test_error() -> None:
    """recording_testのエラーテスト"""
    mock_recorder = Mock()
    mock_recorder.record_and_save.side_effect = ValueError("テストエラー")

    # エラーが発生してもクラッシュしないことを確認
    recording_test(mock_recorder)


def test_transcribe_existing_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """transcribe_existing_fileのファイル未検出テスト"""
    inputs = iter(["nonexistent.wav"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()

    transcribe_existing_file(mock_transcriber)

    # ファイルが見つからない場合、transcribeは呼ばれない
    mock_transcriber.transcribe.assert_not_called()


def test_transcribe_existing_file_success(
    sample_audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe_existing_fileの成功テスト"""
    inputs = iter([str(sample_audio_file), "", ""])  # パス、タイトル、コンテキスト
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = ("テスト文字起こし", [])
    mock_transcriber.save_transcript.return_value = Path("/tmp/transcript.txt")

    transcribe_existing_file(mock_transcriber)

    mock_transcriber.transcribe.assert_called_once()
    mock_transcriber.save_transcript.assert_called_once()


def test_transcribe_existing_file_with_minutes(
    sample_audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe_existing_fileの議事録付きテスト"""
    inputs = iter([str(sample_audio_file), "会議", "参加者"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = ("テスト文字起こし", [])
    mock_transcriber.save_transcript.return_value = Path("/tmp/transcript.txt")

    mock_minutes_gen = Mock()
    mock_minutes_gen.generate_and_save.return_value = ("議事録", Path("/tmp/minutes.md"))

    transcribe_existing_file(mock_transcriber, mock_minutes_gen)

    mock_minutes_gen.generate_and_save.assert_called_once()


def test_generate_minutes_from_transcript_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_minutes_from_transcriptのファイル未検出テスト"""
    inputs = iter(["nonexistent.txt"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_minutes_gen = Mock()

    generate_minutes_from_transcript(mock_minutes_gen)

    # ファイルが見つからない場合、generateは呼ばれない
    mock_minutes_gen.generate_and_save.assert_not_called()


def test_generate_minutes_from_transcript_success(
    sample_transcript_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_minutes_from_transcriptの成功テスト"""
    inputs = iter([str(sample_transcript_file), "会議タイトル", "参加者情報"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_minutes_gen = Mock()
    mock_minutes_gen.generate_and_save.return_value = ("議事録", Path("/tmp/minutes.md"))

    generate_minutes_from_transcript(mock_minutes_gen)

    mock_minutes_gen.generate_and_save.assert_called_once()
    call_args = mock_minutes_gen.generate_and_save.call_args
    assert call_args.kwargs["meeting_title"] == "会議タイトル"
    assert call_args.kwargs["additional_context"] == "参加者情報"


def test_generate_minutes_error(
    sample_transcript_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate_minutes_from_transcriptのエラーテスト"""
    inputs = iter([str(sample_transcript_file), "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_minutes_gen = Mock()
    mock_minutes_gen.generate_and_save.side_effect = ValueError("テストエラー")

    # エラーが発生してもクラッシュしないことを確認
    generate_minutes_from_transcript(mock_minutes_gen)


def test_transcribe_file_error(sample_audio_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """transcribe_existing_fileの文字起こしエラーテスト"""
    inputs = iter([str(sample_audio_file), "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.side_effect = ValueError("テストエラー")

    transcribe_existing_file(mock_transcriber)

    mock_transcriber.save_transcript.assert_not_called()


def test_transcribe_file_minutes_error(
    sample_audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe_existing_fileの議事録エラーテスト"""
    inputs = iter([str(sample_audio_file), "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = ("テスト", [])
    mock_transcriber.save_transcript.return_value = Path("/tmp/transcript.txt")

    mock_minutes_gen = Mock()
    mock_minutes_gen.generate_and_save.side_effect = ValueError("エラー")

    transcribe_existing_file(mock_transcriber, mock_minutes_gen)

    # 文字起こしは成功しているはず
    mock_transcriber.transcribe.assert_called_once()


def test_generate_minutes_invalid_file_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """generate_minutes_from_transcriptの不正なファイル拡張子テスト"""
    # WAVファイルを作成
    wav_file = tmp_path / "test.wav"
    wav_file.write_bytes(b"RIFF" + b"\x00" * 40)

    inputs = iter([str(wav_file)])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_minutes_gen = Mock()

    generate_minutes_from_transcript(mock_minutes_gen)

    # エラーメッセージが出力されることを確認
    assert "サポートされていないファイル形式" in caplog.text
    assert "メニュー2を選択" in caplog.text
    mock_minutes_gen.generate_and_save.assert_not_called()


def test_generate_minutes_unicode_decode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """generate_minutes_from_transcriptのUnicodeDecodeErrorテスト"""
    # バイナリデータを含むテキストファイルを作成
    text_file = tmp_path / "test.txt"
    text_file.write_bytes(b"\x91\x92\x93\x94")

    inputs = iter([str(text_file)])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_minutes_gen = Mock()

    generate_minutes_from_transcript(mock_minutes_gen)

    # エラーメッセージが出力されることを確認
    assert "ファイルの読み込みに失敗" in caplog.text
    assert "UTF-8エンコーディングエラー" in caplog.text
    mock_minutes_gen.generate_and_save.assert_not_called()


def test_transcribe_existing_file_invalid_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """transcribe_existing_fileの不正なファイル拡張子テスト"""
    # テキストファイルを作成
    text_file = tmp_path / "test.txt"
    text_file.write_text("これはテキストファイルです", encoding="utf-8")

    inputs = iter([str(text_file)])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()

    transcribe_existing_file(mock_transcriber)

    # エラーメッセージが出力されることを確認
    assert "サポートされていないファイル形式" in caplog.text
    assert "メニュー3を選択" in caplog.text
    mock_transcriber.transcribe.assert_not_called()


def test_transcribe_existing_file_with_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe_existing_fileの引用符付きパステスト"""
    # 特殊文字を含むファイル名を作成
    audio_file = tmp_path / "【テスト】ファイル名 with spaces.mp3"
    audio_file.write_bytes(b"dummy audio data")

    # 引用符付きのパスを入力
    quoted_path = f'"{str(audio_file)}"'
    inputs = iter([quoted_path, "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_transcriber = Mock()
    mock_transcriber.transcribe.return_value = ("テスト", [])
    mock_transcriber.save_transcript.return_value = tmp_path / "transcript.txt"

    transcribe_existing_file(mock_transcriber)

    # 文字起こしが呼ばれたことを確認
    mock_transcriber.transcribe.assert_called_once()
    # 引数のパスが正しいことを確認
    call_args = mock_transcriber.transcribe.call_args[0][0]
    assert call_args == audio_file


def test_generate_minutes_with_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_minutes_from_transcriptの引用符付きパステスト"""
    # 特殊文字を含むファイル名を作成
    transcript_file = tmp_path / "【議事録】会議 メモ.txt"
    transcript_file.write_text("会議の内容です", encoding="utf-8")

    # シングルクォート付きのパスを入力
    quoted_path = f"'{str(transcript_file)}'"
    inputs = iter([quoted_path, "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    mock_minutes_gen = Mock()
    mock_minutes_gen.generate_and_save.return_value = (
        "議事録",
        tmp_path / "minutes.md",
    )

    generate_minutes_from_transcript(mock_minutes_gen)

    # 議事録生成が呼ばれたことを確認
    mock_minutes_gen.generate_and_save.assert_called_once()
    # transcriptの内容が正しく読み込まれたことを確認
    call_args = mock_minutes_gen.generate_and_save.call_args[0][0]
    assert "会議の内容" in call_args


def test_transcribe_existing_file_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transcribe_existing_fileのチルダ展開テスト"""
    # ホームディレクトリ内に一時ファイルを作成
    from pathlib import Path as PathLib

    home = PathLib.home()
    test_file = home / "test_audio.mp3"

    try:
        test_file.write_bytes(b"dummy audio")

        # チルダを使ったパスを入力
        tilde_path = "~/test_audio.mp3"
        inputs = iter([tilde_path, "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        mock_transcriber = Mock()
        mock_transcriber.transcribe.return_value = ("テスト", [])
        mock_transcriber.save_transcript.return_value = tmp_path / "transcript.txt"

        transcribe_existing_file(mock_transcriber)

        # 文字起こしが呼ばれたことを確認
        mock_transcriber.transcribe.assert_called_once()
    finally:
        # クリーンアップ
        if test_file.exists():
            test_file.unlink()
