"""ロギング設定モジュール"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    ロガーをセットアップ

    Args:
        name: ロガー名
        level: ログレベル

    Returns:
        設定されたロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既にハンドラが設定されている場合はスキップ
    if logger.handlers:
        return logger

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # フォーマッター（絵文字を使って視覚的に）
    formatter = logging.Formatter(
        "%(message)s"  # シンプルなフォーマット（絵文字が既にメッセージに含まれる）
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def setup_file_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    ファイル出力も含むロガーをセットアップ

    Args:
        name: ロガー名
        log_file: ログファイルのパス
        level: ログレベル

    Returns:
        設定されたロガー
    """
    logger = setup_logger(name, level)

    # ファイルハンドラを追加
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    # ファイル用フォーマッター（タイムスタンプ付き）
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    return logger
