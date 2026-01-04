# ragas評価（eval/evaluate.py）

このディレクトリの `eval/evaluate.py` は、`eval_runs/*.jsonl` に溜まった LLM 出力を `ragas` でスコアリングし、`eval_reports/` にレポートを書き出します。

## 前提

- Python 3.10+ 推奨
- リポジトリ直下に `.env`（評価用のAPIキー）

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
```

## APIキー

どちらかを用意します。

- OpenAI: `OPENAI_API_KEY`
- Google Gemini: `GOOGLE_API_KEY`（または `GEMINI_API_KEY`）

補足:
- `eval/evaluate.py` はデフォルトでリポジトリ直下の `.env` を読み込みます（実行ディレクトリに依存しません）。
- 別の dotenv を使いたい場合は `--env-file` を指定します。

## 使い方（基本）

### ニュース取得（news_report）を評価

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report
```

### 差分要約（diff_summary）を評価

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task diff_summary
```

## 入力の指定方法

`--input` は次を受け付けます。

- ディレクトリ: `eval_runs`
- 単一ファイル: `eval_runs/xxx.jsonl`
- glob: `eval_runs/*.jsonl`

## 出力形式（読みやすさ重視）

### デフォルト: Markdown（おすすめ）

- `eval_reports/ragas_YYYYMMDD_HHMMSS.md`
- 長い `response` はプレビュー＋折りたたみ（全文/contexts）で表示

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --out-format md
```

プレビューの長さは `--preview-chars` で調整できます。

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --out-format md --preview-chars 400
```

### JSONL

後処理（集計/フィルタ）しやすい形式です。

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --out-format jsonl
```

### CSV（従来）

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --out-format csv
```

## 出力される産物の読み方

### 1) Markdown（`.md`）

`eval_reports/ragas_YYYYMMDD_HHMMSS.md` は、人間が読む用のレポートです。

- 先頭に **平均スコア**（メトリクスごとの平均）が出ます
- 以降にサンプルが並び、各サンプルは以下で構成されます
	- **Question**: 評価対象の入力（= `user_input`）
	- **Answer (preview)**: `response` の冒頭プレビュー（長文でも見やすい）
	- 折りたたみ（details）: `retrieved_contexts` と `response` の全文

サンプル見出しの末尾に `answer_relevancy=...` のようなスコアが付くことがあります（数値カラムのみ表示）。

### 2) JSONL（`.jsonl`）

後処理（集計/フィルタ/可視化）に向いた機械可読な形式です。
1行=1サンプルで、主に以下のキーを含みます。

- `user_input`: 質問（元の `question`）
- `response`: 回答（元の `answer`）
- `retrieved_contexts`: コンテキスト（元の `contexts`）
- `answer_relevancy`: 回答の関連度スコア（0〜1、1が良い）
- `faithfulness`: コンテキストがある場合のみ算出されることがあります

### 3) CSV（`.csv`）

列は JSONL と同様ですが、`response` が長くなりやすく閲覧性が落ちます。
スプレッドシートでの閲覧より、JSONL→加工 or Markdown を推奨します。

### NaN（数値が NaN になる）ケース

- 評価用LLMが失敗した（レート制限/認証/モデル不一致など）
- `retrieved_contexts` が空で、メトリクス条件を満たしていない
- 出力が期待形式から外れていて、採点が成立しなかった

まずは `eval/evaluate.py` の実行ログ（例外）を確認してください。

## 数値（スコア）を改善する方法

スコアは「評価用LLM（judge）」と「埋め込み（embeddings）」にも依存しますが、最も効くのは **評価対象の出力（news/diffの本文）を改善すること** です。

### answer_relevancy を上げる

`answer_relevancy` は「質問に対して回答がどれだけズレていないか」を見ます。

- 質問の要求を先頭で満たす（結論→根拠の順）
- 余計な前置き（"調査しました" など）を減らす
- 固有名詞/日付/数値など、質問の核になる情報を落とさない
- 形式を安定させる（例: 箇条書き3点、重要度順、など）

### faithfulness を上げる（= “根拠付き” にする）

`faithfulness` は `retrieved_contexts`（根拠）に基づいて回答が書けているかを見ます。
つまり、**コンテキストをちゃんと付ける** のが前提になります。

- `contexts` にニュース本文/IR本文の該当箇所（抜粋）を入れる
- 回答内に「根拠にある事実」だけを書く（推測で埋めない）
- 1つの主張に対して根拠がある状態を保つ（出典の混在を避ける）

特に `diff_summary` は `contexts=[旧版, 新版]` を与えると評価しやすくなります。

### judge / embeddings を変えて安定させる

同じ出力でも、評価用モデルや埋め込みモデルでブレます。

- `--provider openai` の方が安定する場合があります（環境により）
- `--llm-model` / `--embedding-model` を明示して比較実験する

例:

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --provider openai \
	--llm-model gpt-4o-mini --embedding-model text-embedding-3-small
```

### 失敗を減らす（= NaN を減らす）

- 実行ログにモデルNOT_FOUNDや認証エラーが出ていないか確認
- `--env-file` で意図したdotenvを読めているか確認
- 入力JSONLが壊れていないか（改行/JSON不正/空行）を確認

## 出力先の指定

`--out` でパスを固定できます（拡張子が `.md/.jsonl/.csv` の場合はそれに追従します）。

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --out eval_reports/my_report.md
```

## プロバイダの指定

通常は環境変数から自動推定しますが、明示したい場合は `--provider` を使います。

```bash
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --provider google
./.venv/bin/python eval/evaluate.py --input eval_runs --task news_report --provider openai
```

モデルを変えたい場合は `--llm-model` / `--embedding-model`（または `RAGAS_*` 環境変数）を使います。

## CLIオプション一覧

`--help` を参照してください。

```bash
./.venv/bin/python eval/evaluate.py --help
```
