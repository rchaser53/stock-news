import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _collect_inputs(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
    # allow globs
    if any(ch in input_path for ch in ["*", "?", "["]):
        return sorted(glob.glob(input_path))
    return [input_path]


@dataclass
class Providers:
    llm: Any
    embeddings: Optional[Any]


def _build_providers(provider: str, llm_model: Optional[str], embedding_model: Optional[str]) -> Providers:
    provider = provider.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=llm_model or os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        embeddings = OpenAIEmbeddings(model=embedding_model or os.getenv("RAGAS_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        return Providers(llm=llm, embeddings=embeddings)

    if provider in ("google", "gemini"):
        # Uses GOOGLE_API_KEY (recommended) or GEMINI_API_KEY if your env is set that way.
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(model=llm_model or os.getenv("RAGAS_GEMINI_MODEL", "gemini-1.5-flash"), temperature=0)
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model or os.getenv("RAGAS_GEMINI_EMBEDDING_MODEL", "text-embedding-004"))
        return Providers(llm=llm, embeddings=embeddings)

    raise ValueError(f"Unsupported provider: {provider}")


def _infer_provider(cli_provider: Optional[str]) -> str:
    if cli_provider:
        return cli_provider
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "google"
    raise RuntimeError("No provider configured. Set OPENAI_API_KEY or GOOGLE_API_KEY (or GEMINI_API_KEY), or pass --provider.")


def _to_ragas_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # ragas expects columns like: question, answer, contexts(list[str]), ground_truth(optional)
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        normalized.append(
            {
                "id": r.get("id"),
                "task": r.get("task"),
                "question": r.get("question") or "",
                "answer": r.get("answer") or "",
                "contexts": r.get("contexts") or [],
                "ground_truth": r.get("ground_truth") or None,
                "meta": r.get("meta") or {},
                "ts": r.get("ts") or None,
            }
        )
    df = pd.DataFrame(normalized)
    if "contexts" in df.columns:
        df["contexts"] = df["contexts"].apply(lambda x: x if isinstance(x, list) else [])
    return df


def main() -> None:
    load_dotenv(override=False)

    # Convenience: many repos set GEMINI_API_KEY, but langchain-google-genai prefers GOOGLE_API_KEY.
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    parser = argparse.ArgumentParser(description="Evaluate stock-news outputs using ragas")
    parser.add_argument("--input", required=True, help="JSONL file, glob, or directory (e.g. eval_runs)")
    parser.add_argument("--task", default=None, help="Filter by task (e.g. diff_summary)")
    parser.add_argument("--provider", default=None, choices=["openai", "google"], help="LLM/embeddings provider")
    parser.add_argument("--llm-model", default=None, help="Override judge LLM model")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model")
    parser.add_argument("--out", default=None, help="Output CSV path (default: eval_reports/ragas_<timestamp>.csv)")
    args = parser.parse_args()

    input_files = _collect_inputs(args.input)
    if not input_files:
        raise SystemExit("No input files found")

    rows: List[Dict[str, Any]] = []
    for p in input_files:
        rows.extend(_read_jsonl(p))

    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]

    if not rows:
        raise SystemExit("No rows to evaluate after filtering")

    df = _to_ragas_frame(rows)

    provider = _infer_provider(args.provider)
    providers = _build_providers(provider, args.llm_model, args.embedding_model)

    # Import ragas lazily (so `--help` doesn't require deps).
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    # Choose metrics based on availability of contexts.
    has_contexts = df["contexts"].apply(lambda x: isinstance(x, list) and len(x) > 0).any()
    metrics = [answer_relevancy]
    if has_contexts:
        metrics.append(faithfulness)

    ragas_ds = Dataset.from_pandas(df[["question", "answer", "contexts"]].copy(), preserve_index=False)

    result = evaluate(
        ragas_ds,
        metrics=metrics,
        llm=providers.llm,
        embeddings=providers.embeddings,
        raise_exceptions=False,
    )

    out_path = args.out
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("eval_reports", f"ragas_{ts}.csv")

    scores_df = result.to_pandas()
    scores_df.to_csv(out_path, index=False)

    summary = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    print("Saved:", out_path)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
