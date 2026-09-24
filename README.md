# meeting-transcriber

完全ローカルで動作する日本語向け会議文字起こしと議事録自動生成ツールです。音声の文字起こしはfaster-whisperが担い、議事録の作成はOllama上のローカルLLMが担うため、データを外部に送らずに会議を記録できます。

## 特徴

- 初回セットアップで依存パッケージとモデルを取得した後は、インターネット接続なしで動作します
- faster-whisperで日本語音声を文字起こしできます
- Ollamaの任意のローカルLLMで議事録を自動生成できます
- ナレッジベースを参照して用語を補正するRAG機能を備えています
- ruffとmypyとpytestでコード品質を担保しています
- pre-commitとGitHub Actionsにgitleaksを組み込み、機密情報の混入を継続的に検査しています

## 動作要件

- Python 3.11以上
- macOSまたはLinuxでの動作を想定しています
- 議事録生成機能を使う場合はOllamaを別途インストールしておいてください

## インストール

依存解決にはuvを使います。インストールしていない場合は[公式手順](https://docs.astral.sh/uv/)に従って導入してください。

```bash
git clone https://github.com/okamyuji/meeting-transcriber.git
cd meeting-transcriber

# 本番依存のみ
uv sync

# 開発依存も含める
uv sync --all-groups
```

Ollamaを利用する場合は議事録生成用のモデルと、RAG用の埋め込みモデルを取得しておきます。

```bash
ollama pull qwen2.5:7b
ollama pull mxbai-embed-large
```

## クイックスタート

```bash
uv run python main.py
```

メニューから処理を選びます。

```text
1. 録音 -> 文字起こし -> 議事録作成（フルワークフロー）
2. 既存の音声ファイルから文字起こしと議事録作成
3. 既存の文字起こしから議事録作成
4. 録音のみ
5. 設定（モデル選択など）
6. 終了
```

文字起こしだけを試したい場合は次のコマンドで音声ファイルを指定できます。

```bash
uv run python -m app.transcriber path/to/audio.wav large-v3
```

## Whisperモデルの選択

`app/transcriber.py` の `JAPANESE_MODELS` で内部マッピングを定義しています。

| 設定値 | 実体 | 用途 |
|---|---|---|
| `small` | `small` | 軽量で低スペック環境向けです |
| `medium` | `medium` | バランス重視の標準モデルです |
| `large-v3` | `large-v3` | 完走の安定性を重視する場合に向いています |
| `large-v3-ja` | `kotoba-tech/kotoba-whisper-v1.0-faster` | 日本語特化のCTranslate2版です |

`kotoba-tech/kotoba-whisper-v1.0` はHuggingFace Transformers形式で配布されているため、faster-whisperでは `model.bin` が見つからずロードに失敗します。日本語特化モデルを利用する場合はCTranslate2変換済みの `kotoba-tech/kotoba-whisper-v1.0-faster` を指定してください。

なお、漫才やバラエティ音声を題材にした実機検証では、Kotoba Whisper系（`v1.0-faster` と `v2.0-faster`）で音声の前半数十秒で文字起こしが停止する事象を確認しました。途中停止に遭遇した場合は `large-v3` への切り替えを試してください。

## RAG機能

`data/knowledge/` ディレクトリにMarkdownファイルを置くと、議事録生成のプロンプトに関連知識が自動的に追加されます。埋め込みは初回計算時に `.rag_cache/` にキャッシュされ、2回目以降の起動を高速化します。Ollamaが停止していたなどで埋め込みに失敗したファイルはキャッシュせず、次回起動時に再計算します。

```text
data/knowledge/
  README.md
  terms.md
```

用語集の中身は自由に編集できます。新しい `.md` ファイルを追加するだけで自動的に取り込まれます。

## 開発

リポジトリには以下の品質ゲートを設定しています。

- ruff: lintとフォーマット
- mypy: 静的型検査
- pytest: 単体テスト
- gitleaks: 機密情報の検査
- pre-commit: 上記をコミット前に一括実行
- GitHub Actions: pushとpull requestで同じチェックを実行

ローカルで一括実行する場合は次のコマンドを使います。

```bash
uv run ruff check app/ main.py
uv run ruff format --check app/ main.py
uv run mypy app/ main.py
uv run pytest
uv run pre-commit run --all-files
```

初回はpre-commitフックの有効化を忘れずにおこなってください。

```bash
uv run pre-commit install
```

## ディレクトリ構成

```text
meeting-transcriber/
  app/
    __init__.py
    logger.py
    recorder.py
    transcriber.py
    minutes_generator.py
    rag.py
  data/
    audio/
    transcripts/
    knowledge/
  .github/workflows/ci.yml
  .pre-commit-config.yaml
  main.py
  pyproject.toml
  README.md
```

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| 録音できない | macOSではシステム設定のプライバシー項目でターミナルにマイク利用を許可してください |
| 文字起こしが途中で止まる | `large-v3` に切り替える、`vad_filter=False` を試す、`beam_size` を下げるなどで挙動を確認してください |
| `kotoba-tech/kotoba-whisper-v1.0` のロード失敗 | CTranslate2版の `kotoba-tech/kotoba-whisper-v1.0-faster` を指定してください |
| Ollamaに接続できない | 別ターミナルで `ollama serve` を起動し、必要なモデルが `ollama list` に表示されることを確認してください |
| メモリ不足 | より小さなモデル（`small` や `medium`）を選んでください |
| 長い会議で「コンテキスト上限を超えて切り詰められる可能性があります」と表示される | 議事録生成はプロンプト長に合わせて `num_ctx` を最大32768トークンまで広げます。これを超える文字起こしは末尾が反映されないため、会議を分けて文字起こししてください |

## コストについて

文字起こしと議事録生成はいずれも自分の計算機の上で動きます。外部APIや有償SaaSは経由しないため、ランニングコストは電力とディスクの実費だけです。クラウド型の文字起こしサービスでよくある従量課金は発生しません。

- 録音: ストレージのみ
- 文字起こし: faster-whisperによるローカル計算
- 議事録生成: Ollamaによるローカル計算

## プライバシーとセキュリティ

音声データも文字起こしも議事録もすべてローカルファイルに保存され、外部に送信されません。

| 機能 | 保存先 | 外部送信 |
|---|---|---|
| 録音 | `data/audio/` | なし |
| 文字起こし | `data/transcripts/` | なし |
| 議事録 | `data/transcripts/` | なし |

機密情報を含む打ち合わせや、組織のセキュリティポリシー上クラウドへの送信が難しい録音でも安心して利用できます。なお、初回セットアップ時にfaster-whisperのモデルファイルとOllamaのLLMモデルをダウンロードするため、その時点ではインターネット接続が必要です。

## ライセンス

MITライセンスのもとで配布しています。著作権表記の名義はokamyujiです。
