"""
Compute aggregate evaluation scores for the RAG project.

This script expects a CSV file with the following columns:

- id:        question identifier
- correctness: 0/1 (or NaN) numeric score
- groundedness: 0/1 (or NaN) numeric score
- safe_refusal: 0/1 (or NaN) numeric score, mainly for unanswerable questions
- group:     question category (e.g., "human_bias", "math", "unanswerable", ...)
- mode:      "rag" or "baseline"

It writes a human-readable summary to `evaluation/results/summary.txt`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("evaluation/results/eval_scores.csv")
OUTPUT_PATH = Path("evaluation/results/summary.txt")


def _safe_mean(series: pd.Series) -> float:
    """Return mean ignoring NaNs; return 0.0 if all values are NaN."""
    if series.dropna().empty:
        return 0.0
    return float(series.mean())


def compute_aggregate_scores(df: pd.DataFrame) -> str:
    """Compute overall + per-group metrics and return a formatted text report."""
    lines: list[str] = []

    modes = sorted(df["mode"].dropna().unique())
    groups = sorted(df["group"].dropna().unique())

    lines.append("=== Overall metrics by mode ===")
    for mode in modes:
        sub = df[df["mode"] == mode]
        correctness = _safe_mean(sub["correctness"])
        groundedness = _safe_mean(sub["groundedness"])
        safe_refusal = _safe_mean(sub["safe_refusal"])
        lines.append(
            f"- {mode}: "
            f"correctness = {correctness:.3f}, "
            f"groundedness = {groundedness:.3f}, "
            f"safe_refusal = {safe_refusal:.3f} "
            f"(n = {len(sub)})"
        )
    lines.append("")

    for mode in modes:
        lines.append(f"=== Per-group metrics for mode = {mode} ===")
        sub_mode = df[df["mode"] == mode]
        for group in groups:
            sub = sub_mode[sub_mode["group"] == group]
            if sub.empty:
                continue
            correctness = _safe_mean(sub["correctness"])
            groundedness = _safe_mean(sub["groundedness"])
            safe_refusal = _safe_mean(sub["safe_refusal"])
            lines.append(
                f"- group = {group:15s} | "
                f"correctness = {correctness:.3f}, "
                f"groundedness = {groundedness:.3f}, "
                f"safe_refusal = {safe_refusal:.3f} "
                f"(n = {len(sub)})"
            )
        lines.append("")

    # Simple comparison summary between RAG and baseline
    if set(modes) >= {"rag", "baseline"}:
        rag = df[df["mode"] == "rag"]
        base = df[df["mode"] == "baseline"]
        lines.append("=== RAG vs. Baseline (overall) ===")
        lines.append(
            f"- correctness: rag = {_safe_mean(rag['correctness']):.3f}, "
            f"baseline = {_safe_mean(base['correctness']):.3f}"
        )
        lines.append(
            f"- groundedness: rag = {_safe_mean(rag['groundedness']):.3f}, "
            f"baseline = {_safe_mean(base['groundedness']):.3f}"
        )
        lines.append(
            f"- safe_refusal: rag = {_safe_mean(rag['safe_refusal']):.3f}, "
            f"baseline = {_safe_mean(base['safe_refusal']):.3f}"
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {INPUT_PATH}. "
            "Run the evaluation script that generates `eval_scores.csv` first."
        )

    df = pd.read_csv(INPUT_PATH)
    report = compute_aggregate_scores(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

