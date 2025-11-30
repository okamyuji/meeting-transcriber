"""app.loggerのテスト"""

import logging
from pathlib import Path

from app.logger import setup_file_logger, setup_logger


def test_setup_logger() -> None:
    """setup_loggerの基本テスト"""
    logger = setup_logger("test_logger")

    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_setup_logger_with_level() -> None:
    """setup_loggerのログレベル指定テスト"""
    logger = setup_logger("test_logger_debug", level=logging.DEBUG)

    assert logger.level == logging.DEBUG


def test_setup_file_logger(tmp_path: Path) -> None:
    """setup_file_loggerのテスト"""
    log_file = tmp_path / "test.log"
    logger = setup_file_logger("test_file_logger", log_file)

    assert logger.name == "test_file_logger"
    assert log_file.exists()

    # ログ出力テスト
    logger.info("Test message")
    content = log_file.read_text(encoding="utf-8")
    assert "Test message" in content


def test_setup_logger_idempotent() -> None:
    """setup_loggerの冪等性テスト（複数回呼び出しても問題ない）"""
    logger1 = setup_logger("test_idempotent")
    handler_count1 = len(logger1.handlers)

    logger2 = setup_logger("test_idempotent")
    handler_count2 = len(logger2.handlers)

    assert logger1 is logger2
    assert handler_count1 == handler_count2
