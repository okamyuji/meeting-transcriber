# 🎙️ 会議文字起こし＆議事録作成ツール

**完全ローカル実行**可能な、日本語高精度の会議文字起こし＆議事録自動作成アプリケーションです。

## ✨ 特徴

- **🇯🇵 日本語に特化** - faster-whisper + **Kotoba Whisper V1.0**（日本語特化モデル）で高精度な文字起こし
- **💻 完全ローカル実行** - インターネット不要！すべてローカルで完結
- **🔒 プライバシー保護** - データは外部に送信されません
- **🤖 ローカルLLM活用** - Ollama（Qwen2.5/Llama3.1等）で議事録を自動生成
- **📚 RAG対応** - ナレッジベース（用語集・プロジェクト情報）を参照した議事録作成
- **⚡ CPU動作** - GPUなしでも快適に動作（int8量子化で高速化）
- **🎙️ 簡単録音** - CLIから直接マイク録音が可能
- **📝 多機能** - 録音・文字起こし・議事録作成を一括処理、または個別実行も可能
- **🔍 コード品質** - ruff + mypy + pytest でコード品質を担保

## 🚀 クイックスタート

### 1. インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd meeting-transcriber

# 依存関係のインストール（本番環境）
uv sync

# 開発環境の場合
uv sync --all-groups
```

### 2. Ollamaのセットアップ（議事録作成機能を使う場合）

議事録作成とRAG機能を使用するには、Ollamaが必要です。

#### インストール

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# または公式サイトからダウンロード
https://ollama.com/download
```

**推奨モデルのダウンロード:**

```bash
# 【日本語特化】議事録生成用モデル（最推奨）
ollama pull qwen2.5:7b
# Qwen2.5は多言語対応で日本語の精度が高く、議事録作成に最適

# または他の日本語対応モデル
ollama pull llama3.1:8b    # Meta製、バランス型
ollama pull gemma2:9b       # Google製、高速

# 【必須】RAG機能用の埋め込みモデル
ollama pull mxbai-embed-large
# ナレッジベース（用語集・プロジェクト情報）の検索に使用
```

#### モデル選択のポイント

| モデル | サイズ | メモリ | 日本語精度 | 速度 | 特徴 |
|--------|--------|--------|-----------|------|------|
| **qwen2.5:7b** | 4.7GB | ~8GB | ⭐⭐⭐⭐⭐ | ⚡⚡ | **最推奨**・日本語に強い・議事録の文章が自然 |
| llama3.1:8b | 4.7GB | ~8GB | ⭐⭐⭐⭐ | ⚡⚡ | バランス型・英語も得意 |
| gemma2:9b | 5.4GB | ~10GB | ⭐⭐⭐⭐ | ⚡⚡⚡ | Google製・高速 |

> **💡 ヒント**: メモリが少ない環境では、より小さいモデル（`qwen2.5:3b` など）も選択可能です。
>
> **注意**: Ollamaがなくても、録音と文字起こしは利用できます（文字起こしのみ実行可能）。

### 3. 実行

```bash
# アプリケーションを起動
uv run python main.py
```

## 📖 使い方

アプリケーションを起動すると、以下のメニューが表示されます。

```shell
【メニュー】
  1. 🎙️  録音 → 文字起こし → 議事録作成（フルワークフロー）
  2. 📁 既存の音声ファイルから文字起こし＆議事録作成
  3. 📝 既存の文字起こしから議事録作成
  4. 🎤 録音のみ（テスト用）
  5. 🔧 設定（モデル選択など）
  6. ❌ 終了
```

### 基本的なワークフロー

#### 1️⃣ 会議を録音して議事録を作成

1. メニューから `1` を選択
2. 会議タイトルや参加者情報を入力（省略可）
3. Enterキーを押して録音開始
4. 会議終了後、もう一度Enterキーを押して録音停止
5. 自動的に文字起こし → 議事録作成が実行されます

#### 2️⃣ 既存の音声ファイルを処理

1. メニューから `2` を選択
2. 音声ファイルのパスを入力（例: `data/audio/meeting.wav`）
3. 会議情報を入力（省略可）
4. 自動的に文字起こし → 議事録作成が実行されます

#### 3️⃣ 既存の文字起こしから議事録のみ作成

1. メニューから `3` を選択
2. 文字起こしファイルのパスを入力
3. 議事録が生成されます

### 各モジュールの個別実行

```bash
# 録音のみ
uv run python -m app.recorder

# 文字起こしのみ
uv run python -m app.transcriber <音声ファイルのパス> [モデル名]

# 議事録作成のみ
uv run python -m app.minutes_generator <文字起こしファイルのパス>
```

## 🔧 設定

### モデル選択

メニューの `5` から、Whisperモデルのサイズを選択できます。

| モデル | メモリ | 速度 | 精度 | 推奨環境 |
|--------|--------|------|------|----------|
| small | ~1GB | ⚡⚡⚡ | ⭐⭐ | 低スペックPC |
| medium | ~2GB | ⚡⚡ | ⭐⭐⭐ | 標準PC |
| large-v3 | ~4GB | ⚡ | ⭐⭐⭐⭐ | 高スペックPC |
| **large-v3-ja** | ~4GB | ⚡ | ⭐⭐⭐⭐⭐ | **日本語特化（推奨）** |

> **💡 推奨**: 日本語の会議には `large-v3-ja`（Kotoba Whisper V1.0）が最適です。
> OpenAI Whisper large-v3 の日本語特化版で、CER/WERが大幅に改善されています。

### 処理時間の目安

CPUの性能によりますが、一般的な目安は以下の通りです：

| 音声長 | small | medium | large-v3 | large-v3-ja |
|--------|-------|--------|----------|-------------|
| 5分 | ~30秒 | ~1分 | ~2分 | ~1.5分 |
| 10分 | ~1分 | ~2分 | ~5分 | ~3分 |
| 30分 | ~3分 | ~6分 | ~15分 | ~10分 |
| 60分 | ~6分 | ~12分 | ~30分 | ~20分 |

> **💡 ヒント**: `large-v3-ja` は large-v3 と同等の精度で、日本語の処理が高速化されています。

### 日本語特化モデルについて

#### Kotoba Whisper V1.0 の特徴

`large-v3-ja` オプションで使用される [kotoba-tech/kotoba-whisper-v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0) は：

- **ベース**: OpenAI Whisper large-v3
- **最適化**: 日本語ASR用に蒸留・チューニング
- **性能向上**:
    - 元の large-v3 より **最大6.3倍高速**
    - **CER（文字誤り率）/WER（単語誤り率）が改善**
    - 日本語の話し言葉により強い

#### 使い方

1. メニューから `5. 🔧 設定` を選択
2. `large-v3-ja` を選択
3. 初回実行時に自動ダウンロード（~4GB）

または、コマンドラインで直接指定：

```bash
uv run python -m app.transcriber <音声ファイル> large-v3-ja
```

## 🧪 開発者向け情報

### コード品質チェック

このプロジェクトは[uvとRuffで実現する高速で堅牢なPython開発環境の構築](https://zenn.dev/okamyuji/articles/uv-practical-guide)に従って構築されています。

```bash
# Ruffでリント＆フォーマット
uv run ruff check app/ main.py --fix
uv run ruff format app/ main.py

# mypyで型チェック
uv run mypy app/ main.py

# すべてのチェックを実行
uv run pre-commit run --all-files
```

### テスト

```bash
# すべてのテストを実行
uv run pytest

# カバレッジレポート付き
uv run pytest --cov=app --cov-report=html

# カバレッジレポートを表示
open htmlcov/index.html
```

### pre-commitフック

開発時は、コミット前に自動的にコード品質チェックが実行されます。

```bash
# pre-commitのインストール
uv run pre-commit install

# 手動実行
uv run pre-commit run --all-files
```

### プロジェクト構造

```shell
meeting-transcriber/
├── app/
│   ├── __init__.py           # パッケージ初期化
│   ├── logger.py             # ロギング設定
│   ├── recorder.py           # 音声録音モジュール
│   ├── transcriber.py        # 文字起こしモジュール
│   ├── minutes_generator.py  # 議事録生成モジュール
│   └── rag.py                # RAG（ナレッジベース）モジュール
├── tests/                    # テストコード
├── data/
│   ├── audio/                # 録音ファイル保存先
│   ├── transcripts/          # 文字起こし・議事録保存先
│   └── knowledge/            # RAG用ナレッジベース（Markdown）
├── .rag_cache/               # 埋め込みキャッシュ（自動生成）
├── main.py                   # メインCLI
├── pyproject.toml            # プロジェクト設定（ruff/mypy/pytest設定含む）
├── .pre-commit-config.yaml   # pre-commit設定
├── .env.example              # 環境変数テンプレート
└── README.md                 # このファイル
```

## 🛠️ トラブルシューティング

### 録音ができない

**macOS の場合:**

システム環境設定 > セキュリティとプライバシー > プライバシー > マイク で、ターミナルにマイクへのアクセス許可を与えてください。

**利用可能なデバイスを確認:**

```bash
uv run python -m app.recorder
```

### 文字起こしが遅い

- モデルサイズを小さくする（`medium` → `small`）
- バックグラウンドアプリを閉じてCPUリソースを確保

### 議事録作成でエラーが出る

| エラーメッセージ | 原因 | 解決方法 |
|----------------|------|---------|
| `Ollamaサーバーに接続できません` | Ollamaが起動していない | `ollama serve` でOllamaを起動 |
| `モデルが見つかりません` | モデルがダウンロードされていない | `ollama pull qwen2.5:7b` でモデルをダウンロード |
| `Connection refused` | Ollamaが起動していない | [Ollama公式サイト](https://ollama.com/download)からインストール |

### メモリ不足エラー

モデルサイズを小さくしてください：

- 最低メモリ要件:
    - small: 4GB RAM
    - medium: 8GB RAM
    - large-v3: 16GB RAM

## 📚 RAG機能（ナレッジベース参照）

RAG（Retrieval-Augmented Generation）機能を使うと、議事録作成時にプロジェクト固有の用語集や情報を自動的に参照できます。

### RAGのセットアップ

1. **埋め込みモデルのダウンロード**

    ```bash
    ollama pull mxbai-embed-large
    ```

2. **ナレッジベースの作成**

    `data/knowledge/` ディレクトリに Markdown ファイルを配置するだけで自動的に参照されます。

    ```bash
    # サンプルファイルはすでに用意されています
    data/knowledge/
    ├── README.md    # 使い方ガイド
    └── terms.md     # サンプル用語集
    ```

3. **用語集の編集**

    `data/knowledge/terms.md` を編集して、プロジェクト固有の情報を追加：

    ```markdown
    # プロジェクト情報

    ## 社内プロジェクト
    - Project Alpha: 新製品開発プロジェクト。2025年Q2リリース予定
    - Project Beta: 既存システムのリニューアル

    ## 技術用語
    - RAG: Retrieval-Augmented Generation（検索拡張生成）
    - LLM: Large Language Model（大規模言語モデル）

    ## チーム構成
    - 開発チーム: 田中さん、佐藤さん
    - 企画チーム: 鈴木さん、高橋さん
    ```

4. **新しいナレッジファイルの追加**

    任意の `.md` ファイルを `data/knowledge/` に追加できます：

    ```bash
    # 例: プロジェクト情報を追加
    echo "# Project Alpha\n\n新製品開発プロジェクト..." > data/knowledge/project_alpha.md
    ```

### RAGの仕組み

1. **自動検索**: 文字起こしテキストから関連するキーワードを抽出
2. **類似度計算**: 埋め込みベクトルのコサイン類似度で関連知識を検索
3. **自動参照**: 上位3件の関連情報を議事録作成時のプロンプトに追加
4. **キャッシュ**: 埋め込みは `.rag_cache/` にキャッシュされ、2回目以降は高速

### ベストプラクティス

- **具体的に書く**: 「新プロジェクト」より「Project Alpha（新製品開発）」
- **略語を展開**: 「RAG（検索拡張生成）」のように正式名称も記載
- **セクション分け**: 見出し（`#`）を使って構造化する
- **定期的に更新**: プロジェクト情報は最新に保つ

### キャッシュのクリア

ナレッジベースが正しく更新されない場合：

```bash
# キャッシュを削除して再起動
rm -rf .rag_cache/
uv run python main.py
```

## 💰 コストについて

### すべて完全無料

- ✅ **録音**: 無料（ストレージのみ）
- ✅ **文字起こし（faster-whisper）**: 無料（CPU/メモリのみ）
- ✅ **議事録作成（Ollama）**: 無料（CPU/メモリのみ）

**外部APIは一切使用しません** - すべてローカルで完結するため、ランニングコストはゼロです。

## 🔒 プライバシーとセキュリティ

### 完全ローカル実行 - データは外部に送信されません

| 機能 | データの保存場所 | 外部送信 |
|-----|-------------|---------|
| 録音 | `data/audio/` (ローカル) | ❌ なし |
| 文字起こし | `data/transcripts/` (ローカル) | ❌ なし |
| 議事録作成 | `data/transcripts/` (ローカル) | ❌ なし |

**すべての処理がローカルで完結** - インターネット接続不要で、機密情報も安全に扱えます。

## 🤝 技術スタック

- **音声録音**: [sounddevice](https://python-sounddevice.readthedocs.io/)
- **文字起こし**:
    - [faster-whisper](https://github.com/guillaumekln/faster-whisper) (CTranslate2ベース高速実装)
    - [Kotoba Whisper V1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0) (日本語特化モデル・推奨)
- **議事録生成**: [Ollama](https://ollama.com/) (ローカルLLM)
- **RAG**: [Ollama Embeddings](https://ollama.com/) (mxbai-embed-large)
- **パッケージ管理**: [uv](https://github.com/astral-sh/uv)
- **コード品質**: [ruff](https://github.com/astral-sh/ruff), [mypy](https://github.com/python/mypy)
- **テスト**: [pytest](https://docs.pytest.org/)

## 📚 参考リンク

### ASR（文字起こし）

- [faster-whisper 公式ドキュメント](https://github.com/guillaumekln/faster-whisper)
- [Kotoba Whisper V1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0) - 日本語特化モデル
- [OpenAI Whisper](https://github.com/openai/whisper) - オリジナル実装

### LLM・RAG

- [Ollama 公式サイト](https://ollama.com/)
- [Ollama モデルライブラリ](https://ollama.com/library)

### 開発ツール

- [uv 公式ドキュメント](https://docs.astral.sh/uv/)
- [uvとRuffで実現する高速で堅牢なPython開発環境の構築](https://zenn.dev/okamyuji/articles/uv-practical-guide)

## 📄 ライセンス

MIT License
