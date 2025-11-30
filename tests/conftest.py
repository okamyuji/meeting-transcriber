"""pytest設定とフィクスチャ"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """一時的な音声ディレクトリ"""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def temp_transcript_dir(tmp_path: Path) -> Path:
    """一時的な文字起こしディレクトリ"""
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    return transcript_dir


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """サンプル音声データ（1秒、16kHz）"""
    sample_rate = 16000
    duration = 1.0
    return np.random.rand(int(sample_rate * duration)).astype(np.float32)


@pytest.fixture
def sample_audio_file(temp_audio_dir: Path, sample_audio_data: np.ndarray) -> Path:
    """サンプル音声ファイル"""
    import soundfile as sf

    audio_path = temp_audio_dir / "test.wav"
    sf.write(str(audio_path), sample_audio_data, 16000)
    return audio_path


@pytest.fixture
def sample_transcript() -> str:
    """サンプル文字起こしテキスト"""
    return """こんにちは、今日の会議を始めます。
まず、先週の進捗について報告をお願いします。
ありがとうございます。それでは次の議題に移ります。
"""


@pytest.fixture
def sample_transcript_file(temp_transcript_dir: Path, sample_transcript: str) -> Path:
    """サンプル文字起こしファイル"""
    transcript_path = temp_transcript_dir / "test_transcript.txt"
    transcript_path.write_text(sample_transcript, encoding="utf-8")
    return transcript_path


@pytest.fixture
def mock_whisper_model() -> Mock:
    """WhisperModelのモック"""
    mock_model = Mock()

    # transcribeメソッドのモック
    mock_segment = Mock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "テストテキスト"

    mock_info = Mock()
    mock_info.duration = 1.0

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    return mock_model


@pytest.fixture
def mock_ollama_response() -> dict:
    """Ollama レスポンスのモック"""
    return {"message": {"content": "# 議事録\n\nテスト議事録です。"}}


@pytest.fixture
def mock_knowledge_base() -> Mock:
    """KnowledgeBaseのモック"""
    mock_kb = Mock()
    mock_kb.search.return_value = "**テスト用語**\nRAG: Retrieval-Augmented Generation"
    return mock_kb
