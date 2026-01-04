import argparse
import glob
import json
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ModuleNotFoundError as e:
    if e.name != "pandas":
        raise
    raise SystemExit(
        "Missing dependency: pandas\n\n"
        "Run the following from repo root:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r eval/requirements.txt\n\n"
        "Then re-run:\n"
        "  python eval/evaluate.py --input eval_runs\n"
    )
from dotenv import load_dotenv


def _repo_root() -> Path:
    # eval/evaluate.py -> repo root is two levels up
    return Path(__file__).resolve().parents[1]


def _resolve_env_file(env_file: Optional[str]) -> Path:
    root = _repo_root()
    if not env_file:
        return root / ".env"
    p = Path(env_file)
    return p if p.is_absolute() else (root / p)


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


def _compat_embeddings(embeddings: Any) -> Any:
    if embeddings is None:
        return None
    if hasattr(embeddings, "embed_query") and hasattr(embeddings, "embed_documents"):
        return embeddings

    class _Compat:
        def __init__(self, inner: Any):
            self._inner = inner

        def embed_query(self, text: str) -> List[float]:
            return self._inner.embed_text(text)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._inner.embed_texts(texts)

    return _Compat(embeddings)


def _build_providers(provider: str, llm_model: Optional[str], embedding_model: Optional[str]) -> Providers:
    provider = provider.lower().strip()

    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)
        model = llm_model or os.getenv("RAGAS_OPENAI_MODEL", "gpt-4o-mini")
        emb_model = embedding_model or os.getenv("RAGAS_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        llm = llm_factory(model=model, provider="openai", client=client, temperature=0)
        embeddings = _compat_embeddings(embedding_factory(provider="openai", model=emb_model, client=client))
        return Providers(llm=llm, embeddings=embeddings)

    if provider in ("google", "gemini"):
        # ragas 0.4+ uses LiteLLM adapter for Gemini.
        # We build an Instructor client over LiteLLM so ragas can request structured outputs.
        try:
            import instructor
            import litellm
        except ModuleNotFoundError:
            raise SystemExit(
                "Missing dependency: litellm/instructor\n\n"
                "Run:\n"
                "  .venv/bin/pip install -r eval/requirements.txt\n"
            )

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")

        model = llm_model or os.getenv("RAGAS_GEMINI_MODEL", "gemini-2.0-flash")
        # For LiteLLM, prefixing with "gemini/" helps select Google AI Studio
        # (API key) instead of Vertex AI (ADC).
        if "/" not in model and not model.startswith("models/"):
            model = f"gemini/{model}"
        emb_model = embedding_model or os.getenv("RAGAS_GEMINI_EMBEDDING_MODEL", "text-embedding-004")

        # LiteLLM commonly looks for GEMINI_API_KEY.
        os.environ.setdefault("GEMINI_API_KEY", api_key)
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

        llm_client = instructor.from_litellm(litellm.completion)
        llm = llm_factory(model=model, provider="google", client=llm_client, adapter="litellm", temperature=0)

        # Embeddings: use GoogleEmbeddings (google-genai SDK) via ragas embedding_factory.
        try:
            from google import genai  # type: ignore[attr-defined]

            emb_client = genai.Client(api_key=api_key)
            embeddings = _compat_embeddings(embedding_factory(provider="google", model=emb_model, client=emb_client))
        except Exception:
            embeddings = _compat_embeddings(embedding_factory(provider="google", model=emb_model))
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
    # We'll normalize to internal columns and later map to ragas expected:
    # user_input, response, retrieved_contexts
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
    parser = argparse.ArgumentParser(description="Evaluate stock-news outputs using ragas")
    parser.add_argument("--input", required=True, help="JSONL file, glob, or directory (e.g. eval_runs)")
    parser.add_argument("--task", default=None, help="Filter by task (e.g. diff_summary)")
    parser.add_argument("--provider", default=None, choices=["openai", "google"], help="LLM/embeddings provider")
    parser.add_argument("--llm-model", default=None, help="Override judge LLM model")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model")
    parser.add_argument("--out", default=None, help="Output CSV path (default: eval_reports/ragas_<timestamp>.csv)")
    parser.add_argument(
        "--env-file",
        default=None,
        help="dotenv file path. If relative, it is resolved from repo root (default: <repo>/.env)",
    )
    args = parser.parse_args()

    env_path = _resolve_env_file(args.env_file)
    if args.env_file and not env_path.exists():
        raise SystemExit(f"dotenv file not found: {env_path}")
    load_dotenv(dotenv_path=str(env_path), override=False)

    # Convenience: many repos set GEMINI_API_KEY, but langchain-google-genai prefers GOOGLE_API_KEY.
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

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

    ragas_llm = providers.llm
    ragas_embeddings = providers.embeddings

    # Import ragas lazily (so `--help` doesn't require deps).
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    # Choose metrics based on availability of contexts.
    has_contexts = df["contexts"].apply(lambda x: isinstance(x, list) and len(x) > 0).any()
    if ragas_embeddings is None:
        raise SystemExit("Embeddings provider is required for answer_relevancy. Please configure embeddings or switch provider.")

    metrics = [answer_relevancy]
    if has_contexts:
        metrics.append(faithfulness)

    eval_df = df.rename(
        columns={
            "question": "user_input",
            "answer": "response",
            "contexts": "retrieved_contexts",
        }
    )[["user_input", "response", "retrieved_contexts"]].copy()
    ragas_ds = Dataset.from_pandas(eval_df, preserve_index=False)

    result = evaluate(
        ragas_ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    out_path = args.out
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("eval_reports", f"ragas_{ts}.csv")

    scores_df = result.to_pandas()
    scores_df.to_csv(out_path, index=False)

    # ragas 0.4+ returns an EvaluationResult object (not dict-like).
    # We compute mean scores from the per-row result dataframe.
    numeric = scores_df.select_dtypes(include=["number"])
    summary = {k: float(v) for k, v in numeric.mean(numeric_only=True, skipna=True).to_dict().items()}
    print("Saved:", out_path)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
