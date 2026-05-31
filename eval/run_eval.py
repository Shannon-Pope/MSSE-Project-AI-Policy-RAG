import csv
import json
import os
import time
from pathlib import Path
from typing import TypedDict
import sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag_pipeline import answer_question, Citation, Snippet


class EvalResult(TypedDict):
    id: str
    question: str
    gold_answer: str
    expected_source: str
    model_answer: str
    citations: str
    snippets: str
    latency_seconds: float
    groundedness_score: float | str
    citation_accuracy_score: float
    notes: str

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "evaluation_questions.csv"
RESULTS_FILE = EVAL_DIR / "results.csv"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

USE_LLM_JUDGE = True  # set to False to skip groundedness scoring

SOURCE_MAP: dict[str, str | None] = {
    "travel_policy": "Business Travel",
    "expense_policy": "Expense Reimbursement",
    "pto_policy": "Paid Time Off",
    "remote_work_policy": "Hybrid",
    "security_policy": "Information Security",
    "out_of_scope": None,
}


def score_citation_accuracy(
    expected_source: str,
    citations: list[Citation],
    model_answer: str,
) -> float:
    fragment = SOURCE_MAP.get(expected_source)

    if fragment is None:
        return 1.0 if "can only answer" in model_answer.lower() else 0.0

    titles = [c["document_title"] for c in citations]
    return 1.0 if any(fragment in t for t in titles) else 0.0


def score_groundedness(question: str, gold_answer: str, model_answer: str) -> float:
    if not OPENROUTER_API_KEY:
        return -1.0

    prompt = (
        f"You are evaluating a RAG assistant's answer to a company policy question.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Model answer: {model_answer}\n\n"
        f"Score the model answer for factual groundedness:\n"
        f"- 1.0 = Fully correct and grounded in the policy\n"
        f"- 0.5 = Partially correct or contains minor extrapolation\n"
        f"- 0.0 = Incorrect, contradicts policy, or refuses when it should answer\n\n"
        f"Reply with only the number (0.0, 0.5, or 1.0)."
    )

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        return float(text)
    except Exception:
        return -1.0


def calculate_latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}

    sorted_latencies = sorted(latencies)

    def percentile(values: list[float], p: int) -> float:
        index = int(round((p / 100) * (len(values) - 1)))
        return values[index]

    return {
        "p50": percentile(sorted_latencies, 50),
        "p95": percentile(sorted_latencies, 95),
    }


def main() -> None:
    results: list[EvalResult] = []
    latencies: list[float] = []

    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    for row in rows:
        question = row["question"]
        print(f"Evaluating ({row['id']}/{total}): {question}")

        notes = ""
        answer = ""
        citations: list[Citation] = []
        snippets: list[Snippet] = []

        start = time.perf_counter()
        try:
            response = answer_question(question)
            answer = response["answer"]
            citations = response["citations"]
            snippets = response["snippets"]
        except Exception as e:
            notes = f"ERROR: {e}"
        finally:
            latency = round(time.perf_counter() - start, 3)
            latencies.append(latency)

        citation_score = score_citation_accuracy(row["expected_source"], citations, answer)

        groundedness_score: float | str = ""
        if USE_LLM_JUDGE and answer:
            groundedness_score = score_groundedness(question, row["gold_answer"], answer)

        results.append({
            "id": row["id"],
            "question": question,
            "gold_answer": row["gold_answer"],
            "expected_source": row["expected_source"],
            "model_answer": answer,
            "citations": json.dumps(citations),
            "snippets": json.dumps(snippets),
            "latency_seconds": latency,
            "groundedness_score": groundedness_score,
            "citation_accuracy_score": citation_score,
            "notes": notes,
        })

    latency_stats = calculate_latency_stats(latencies)

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "question",
            "gold_answer",
            "expected_source",
            "model_answer",
            "citations",
            "snippets",
            "latency_seconds",
            "groundedness_score",
            "citation_accuracy_score",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    grounded: list[float] = [r["groundedness_score"] for r in results if isinstance(r["groundedness_score"], float) and r["groundedness_score"] >= 0]
    cited: list[float] = [r["citation_accuracy_score"] for r in results]

    print("\nEvaluation complete.")
    print(f"Saved results to: {RESULTS_FILE}")
    print(f"Latency p50: {latency_stats['p50']}s  p95: {latency_stats['p95']}s")
    if grounded:
        print(f"Groundedness avg: {sum(grounded) / len(grounded):.2f}  ({len(grounded)}/{total} scored)")
    if cited:
        print(f"Citation accuracy avg: {sum(cited) / len(cited):.2f}")


if __name__ == "__main__":
    main()
