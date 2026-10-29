"""
Standalone RAGAS evaluation of the RAG pipeline (rag/rag_engine.py).

Run manually — never from tests/CI:

    python -m evaluation.evaluate_rag

Every run makes real LLM calls, both to answer the golden questions (via the
actual production pipeline) and to judge them (RAGAS' own LLM-as-judge
metrics), so it has a real API cost each time.

The golden question set below was written by hand against this project's real
ingested documentation (product names pulled from
sections/catalogue/catalogue_sections.json, so every question actually has an
answer somewhere in the knowledge base). The "reference" field re-uses the
pipeline's OWN generated answer rather than a hand-written expected answer —
faithfulness/answer_relevancy don't need an independent reference at all;
context_precision/context_recall do, and self-referencing means those two
mostly measure "did retrieval find the same context the answer was grounded
in" rather than true recall against an independently verified ground truth.
Review the generated answers in evaluation/last_run.json by hand — that's the
whole point of self-referencing instead of skipping the golden set entirely.
"""

from pathlib import Path

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from backend.llm_provider import get_embeddings, get_llm
from rag.rag_engine import get_assistant_response

GOLDEN_QUESTIONS = [
    # One per real catalogue product (sections/catalogue/catalogue_sections.json)
    "Quali sono le caratteristiche principali del modello IRIS?",
    "Che materiale ha il modello DIAMANT?",
    "Descrivi il modello STILOS.",
    "Che tipo di finestra è il modello EASY?",
    "Quali sono le caratteristiche del modello MINIMAL?",
    "Descrivi il modello MAGIS 40.",
    "Che caratteristiche ha il modello ELEGANT?",
    # Installation procedures (installer/manager-only collection)
    "Quali sono le procedure generali di installazione per finestre e porte?",
    "Come si prepara l'area prima dell'installazione di un infisso?",
    "Quali controlli iniziali vanno fatti prima di installare una finestra?",
    "Come si misura correttamente il vano prima dell'installazione?",
    # Maintenance (public collection)
    "Come si pulisce e mantiene un serramento in PVC?",
    "Come si mantiene un serramento in legno o legno-alluminio?",
    "Che prodotti si usano per la manutenzione del legno?",
    "Come si effettua la manutenzione di una porta blindata?",
    # Edge cases worth tracking over time
    "Qual è la differenza tra i modelli IRIS e DIAMANT?",
    "Un serramento a taglio termico riduce la dispersione di calore?",
    "Quanto costa il modello IRIS?",  # deliberately unanswerable from docs (no pricing in catalogue text) — checks the model doesn't hallucinate a price
]


def build_dataset() -> EvaluationDataset:
    samples = []
    for question in GOLDEN_QUESTIONS:
        answer, docs = get_assistant_response(question)
        contexts = [d.page_content for d in docs] or ["(no context retrieved)"]
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=answer,
            )
        )
        print(f"  answered: {question[:60]}...")
    return EvaluationDataset.from_list([s.model_dump() for s in samples])


def main():
    print(
        f"Building golden dataset ({len(GOLDEN_QUESTIONS)} questions) — this runs the real RAG pipeline for each one:"
    )
    dataset = build_dataset()

    print(
        "\nRunning RAGAS evaluation (faithfulness, answer relevancy, context precision/recall)..."
    )
    # RAGAS' own judge prompts (claim decomposition, multi-step scoring) need
    # more headroom than get_llm's default max_tokens=700 — a truncated judge
    # response raises LLMDidNotFinishException and that sub-score silently
    # comes back as 0, which looks exactly like a real faithfulness failure
    # unless you go check the raw generation. 4000 avoids that.
    llm = LangchainLLMWrapper(get_llm(temperature=0, max_tokens=4000))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    metric_cols = [
        c
        for c in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        if c in df.columns
    ]

    print("\n" + "=" * 60)
    print(f"RAGAS Evaluation Report — {len(GOLDEN_QUESTIONS)} questions")
    print("=" * 60)
    for metric in metric_cols:
        print(f"{metric:20s}: {df[metric].mean():.3f}")
    print("=" * 60)

    if metric_cols:
        worst = df.sort_values(by=metric_cols[0]).head(3)
        print(f"\nLowest-scoring questions (by {metric_cols[0]}):")
        for _, row in worst.iterrows():
            print(f"  [{row[metric_cols[0]]:.2f}] {row['user_input'][:80]}")

    out_path = Path(__file__).parent / "last_run.json"
    df.to_json(out_path, orient="records", indent=2, force_ascii=False)
    print(f"\nFull per-question scores saved to {out_path}")
    print("Review the generated answers/contexts there by hand before trusting this as a baseline.")


if __name__ == "__main__":
    main()
