# 株価ニュース取得プログラム

このプログラムは、設定ファイルに記載された会社の株価に関連する最新ニュースをGemini APIを使用して取得し、テキストファイルとして保存します。また、保存したファイルの内容を音声で読み上げる機能も搭載しています。

## TypeScript(Node.js)版（推奨）

### 必要な環境

- Node.js 18+（推奨 20+）
- Gemini APIキー
- Docker（VOICEVOX Engine用）
- macOS（音声ファイル再生用に`afplay`コマンドを使用）

### セットアップ

```bash
npm install
cp .env.example .env
```

`.env` に Gemini の設定をしてください（最低限 `GEMINI_API_KEY`）。

VOICEVOX Engineを起動:

```bash
docker-compose up -d
```

### 実行方法

ニュース取得（デフォルト）:

```bash
npm run dev
```

情報取得後に自動で差分読み上げ（おすすめ）:

```bash
npm run dev -- -auto-read
```

読み上げモード:

```bash
npm run dev -- -read -dir output/2025-12-06
```

## 設定ファイル

`config.yaml` に監視したい会社のリストを記載します:

```yaml
companies:
  - name: "トヨタ自動車"
    ticker: "7203"
    ir_url: "https://global.toyota/jp/ir/"
```

- `name`: 会社名
- `ticker`: 証券コード
- `ir_url`: IR情報サイトのURL

## 実行方法

### 1. ニュース取得モード（デフォルト）

```bash
npm run dev
```

実行すると、`output/YYYY-MM-DD/` ディレクトリに各会社のニュースがテキストファイルとして保存されます。

### 2. 自動差分読み上げモード（おすすめ）

ニュースを取得した後、直前の最新情報と自動で比較し、変更があったファイルのみを**要約して**音声で読み上げます：

```bash
npm run dev -- -auto-read
```

このモードでは：
- 新規に情報を取得
- 直近の最新ディレクトリ（例: 昨日のデータ）と比較
- 差分がある内容をAIで要約（200文字以内）
- 要約された変更点のみを自動で読み上げ
- 変更がない場合は読み上げをスキップ

**実行例:**
```
=== 株価関連情報 ===
出力ディレクトリ: output/2025-12-06

[1] スポーツフィールド (7080) を処理中...
  ✓ 完了: output/2025-12-06/スポーツフィールド_7080.txt

=== 処理完了 ===

=== 差分チェック ===
新規データ: output/2025-12-06
比較対象: output/2025-12-04

  スポーツフィールド_7080.txt: 差分検出
  グランディハウス_8999.txt: 変更なし
  クオルテック_9165.txt: 差分検出

=== 差分を要約して読み上げます（2件） ===

[1/2] スポーツフィールド_7080.txt の差分を処理中...
  差分を要約中...
  要約完了
  読み上げ中...
  ✓ 完了

[2/2] クオルテック_9165.txt の差分を処理中...
  差分を要約中...
  要約完了
  読み上げ中...
  ✓ 完了

=== 差分読み上げ完了 ===
```

### 3. 音声読み上げモード

指定したディレクトリ内の全ての `.txt` ファイルを音声で読み上げます：

```bash
npm run dev -- -read -dir output/2025-12-06
```

**オプション:**
- `-read`: 音声読み上げモードを有効化
- `-dir <path>`: 読み上げ対象のディレクトリパスを指定
- `-auto-read`: 情報取得後に自動で差分を読み上げる（おすすめ）

**使用例:**
```bash
# 今日のニュースを読み上げ
npm run dev -- -read -dir output/2025-12-06

# 過去のニュースを読み上げ
npm run dev -- -read -dir output/2025-11-15

# 情報取得後、自動で差分のみ読み上げ（最も効率的）
npm run dev -- -auto-read
```

## 出力例

### ニュース取得時
```
=== 株価関連情報 ===
出力ディレクトリ: output/2025-12-06

[1] スポーツフィールド (7080) を処理中...
  ✓ 完了: output/2025-12-06/スポーツフィールド_7080.txt
[2] グランディハウス (8999) を処理中...
  ✓ 完了: output/2025-12-06/グランディハウス_8999.txt

=== 処理完了 ===
```

### 音声読み上げ時
```
=== 音声読み上げ開始 ===
ディレクトリ: output/2025-12-06
対象ファイル数: 7

[1/7] スポーツフィールド_7080.txt を読み上げ中...
  ✓ 完了
[2/7] グランディハウス_8999.txt を読み上げ中...
  ✓ 完了

=== 音声読み上げ完了 ===
```

### 自動差分読み上げ時
```
=== 差分チェック ===
新規データ: output/2025-12-06
比較対象: output/2025-12-04

  スポーツフィールド_7080.txt: 差分検出
  グランディハウス_8999.txt: 変更なし

=== 差分を要約して読み上げます（1件） ===

[1/1] スポーツフィールド_7080.txt の差分を処理中...
  差分を要約中...
  要約完了
  読み上げ中...
  ✓ 完了

=== 差分読み上げ完了 ===
```

**要約例:**
> 「スポーツフィールドに関する変更点です。新規に開示された決算短信により、売上高が前年比15%増加し、営業利益も好調に推移していることが判明しました。株価への好影響が期待されます。」

## 注意事項

- Gemini APIの利用には料金が発生します
  - 情報取得/差分要約: デフォルトでは `gemini-3-pro-preview` を使用します（環境変数で変更可能）
- APIのレート制限にご注意ください
- 音声読み上げ機能は**VOICEVOX Engine**を使用し、**ずんだもん**（speakerID=3）の声で読み上げます
- VOICEVOX Engineは事前に起動しておく必要があります（`docker-compose up -d`）
- 音声ファイルの再生にはmacOSの`afplay`コマンドを使用します
- 差分比較は、ファイルのニュース本文部分のみを対象とします（ヘッダー情報は除外）
- `-auto-read`オプション使用時は、直近の最新ディレクトリと比較します
- 差分要約は200文字以内で、株価に影響する重要な情報を優先します

## ragas による自動評価（LLM出力のスコアリング）

このプロジェクトは TypeScript(Node.js) でニュース取得/差分要約を行い、出力品質の評価には Python ライブラリの `ragas` を利用できます。

### 何が評価できるか

- `news_report`（ニュース取得の出力）: `question` と `answer` を使った評価（例: answer relevancy）
- `diff_summary`（差分要約の出力）: `question` / `answer` に加え、`contexts=[旧版, 新版]` を使った評価（例: faithfulness）

※ `ragas` の多くのメトリクスは「評価用LLM（judge）」と「埋め込み（embeddings）」を必要とします。

### 1) 評価用JSONLの出力

通常実行でも `news_report` を `eval_runs/*.jsonl` に追記します。

```bash
npm run dev
```

差分要約 (`diff_summary`) も評価したい場合は `-auto-read` を使ってください（LLM要約が成功したものだけが記録されます）。

```bash
npm run dev -- -auto-read
```

出力先は環境変数で変更できます:

```bash
# 1実行分のJSONLのパスを固定したい場合
export EVAL_RUN_FILE="eval_runs/my_run.jsonl"

# ディレクトリを変えたい場合（EVAL_RUN_FILE未指定時に使用）
export EVAL_OUTPUT_DIR="eval_runs"
```

### 2) ragas の実行

Python仮想環境を作成して依存関係を入れます:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
```

評価用のAPIキーを用意します（どちらか）:

- OpenAIを使う: `OPENAI_API_KEY`
- Google Geminiを使う: `GOOGLE_API_KEY`（または既存の `GEMINI_API_KEY`）

実行例（差分要約のみ評価）:

```bash
python eval/evaluate.py --input eval_runs --task diff_summary
```

実行例（ニュース取得のみ評価）:

```bash
python eval/evaluate.py --input eval_runs --task news_report
```

結果は `eval_reports/ragas_YYYYMMDD_HHMMSS.csv` に保存され、平均スコアも標準出力に表示されます。

## 機能詳細

### 差分要約機能
`-auto-read`モードでは、AIが以下の処理を行います：
1. 旧版と新版の内容を比較
2. 変更点を箇条書きで3点以内に要約
3. 株価に影響する重要な情報を優先
4. 要約された内容のみを音声で読み上げ

これにより、長文の全文を聞く必要がなく、重要な変更点だけを効率的に把握できます。

## 使用シーン

### 毎日の定期チェック
```bash
# VOICEVOX Engineを起動
docker-compose up -d

# cronやタスクスケジューラで定期実行
npm run dev -- -auto-read
```
毎日実行すれば、前日からの変更点のみをずんだもんの声で確認できます。

### 過去データとの比較
新規取得したデータは日付ごとに保存されるため、過去の情報と比較できます。

## VOICEVOX について

このプログラムは[VOICEVOX Engine](https://github.com/VOICEVOX/voicevox_engine)を使用して、ずんだもん（東北ずん子）の声で音声読み上げを行います。

- **使用音声:** ずんだもん（speakerID=3）
- **Docker Image:** `voicevox/voicevox_engine:cpu-latest`
- **API Endpoint:** `http://localhost:50021`

他のキャラクターを使用したい場合は、[main.go](main.go)の`speakWithVoicevox`関数呼び出し部分で`speakerID`を変更してください。
他のキャラクターを使用したい場合は、[src/voicevox.ts](src/voicevox.ts) の `speakWithVoicevox(text, speakerId)` 呼び出し側の `speakerId` を変更してください。

**主要なspeakerID一覧:**
- 3: ずんだもん（ノーマル）
- 1: 四国めたん（ノーマル）
- 8: 春日部つむぎ（ノーマル）
- 10: 雨晴はう（ノーマル）

詳細は[VOICEVOXの公式ドキュメント](http://localhost:50021/docs)を参照してください（VOICEVOX Engine起動中にアクセス可能）。
