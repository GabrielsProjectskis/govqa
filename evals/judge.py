"""LLM-as-judge prompts, and a check of the judge against human labels.

    uv run python -m evals.judge --export 40   # writes results/human_labels.csv to fill in
    uv run python -m evals.judge --agreement   # Cohen's kappa judge vs human

A judge is only worth trusting once it has been compared with a person, so the README reports
generation scores together with this agreement figure.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from govqa import config
from govqa.llm import complete, parse_json
from govqa.metrics import cohen_kappa

FAITHFULNESS_PROMPT = """Je beoordeelt of een antwoord volledig wordt gedekt door de bronnen.

Bronnen:
{sources}

Antwoord:
{answer}

Controleer elke feitelijke bewering in het antwoord. "faithful" is true als ALLE beweringen in \
de bronnen staan, anders false. Verwijzingen zoals [1] zijn geen beweringen.
Antwoord alleen met JSON: {{"faithful": true|false, "unsupported": "<de niet-gedekte bewering \
of leeg>"}}"""

CORRECTNESS_PROMPT = """Je beoordeelt of een antwoord op een vraag inhoudelijk klopt, vergeleken \
met de officiële tekst.

Vraag: {question}

Officiële tekst:
{reference}

Antwoord om te beoordelen:
{answer}

Score: 2 = beantwoordt de vraag en komt overeen met de officiële tekst; 1 = deels goed of \
onvolledig op een belangrijk punt; 0 = fout, misleidend of beantwoordt de vraag niet.
Antwoord alleen met JSON: {{"score": 0|1|2, "reason": "<kort>"}}"""


def _ask(prompt: str) -> dict:
    msg = [{"role": "user", "content": prompt}]
    return parse_json(complete(msg, model=config.JUDGE_MODEL, max_tokens=300).text)


def judge_faithfulness(answer: str, sources: list[str]) -> dict:
    blocks = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))
    out = _ask(FAITHFULNESS_PROMPT.format(sources=blocks, answer=answer))
    return {"faithful": bool(out.get("faithful")), "unsupported": str(out.get("unsupported", ""))}


def judge_correctness(question: str, answer: str, reference: str) -> dict:
    reference = " ".join(reference.split()[:600])
    out = _ask(CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer))
    return {"score": int(out.get("score", 0)), "reason": str(out.get("reason", ""))}


ROWS_PATH = config.RESULTS_DIR / "generation_rows.jsonl"
LABELS_PATH = config.RESULTS_DIR / "human_labels.csv"


def _load_rows() -> pd.DataFrame:
    rows = [json.loads(line) for line in ROWS_PATH.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    return df[(df.kind == "answerable") & ~df.refused]


def export_sheet(n: int) -> None:
    sample = _load_rows().sample(n=n, random_state=0)
    sheet = sample[["doc_id", "question", "answer", "url"]].copy()
    sheet["human_faithful"] = ""  # 1 = every claim is on the cited page, 0 = not
    sheet["human_correct"] = ""  # 1 = fully correct answer, 0 = partly or not
    sheet.to_csv(LABELS_PATH, index=False, encoding="utf-8-sig")
    print(f"wrote {LABELS_PATH}; fill in the two human_* columns with 0 or 1")


def agreement() -> dict:
    labels = pd.read_csv(LABELS_PATH, encoding="utf-8-sig").dropna(
        subset=["human_faithful", "human_correct"]
    )
    merged = labels.merge(_load_rows(), on="doc_id", suffixes=("", "_run"))
    out = {
        "n": len(merged),
        "kappa_faithful": round(
            cohen_kappa(
                merged.human_faithful.astype(int).tolist(), merged.faithful.astype(int).tolist()
            ),
            3,
        ),
        "kappa_correct": round(
            cohen_kappa(
                merged.human_correct.astype(int).tolist(),
                (merged.correctness == 2).astype(int).tolist(),
            ),
            3,
        ),
    }
    (config.RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", type=int, default=0, metavar="N")
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()
    if args.export:
        export_sheet(args.export)
    if args.agreement:
        print(agreement())


if __name__ == "__main__":
    main()
