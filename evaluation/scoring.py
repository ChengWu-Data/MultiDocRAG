import pandas as pd
import ast
from pathlib import Path

# ============================
#  Configuration
# ============================
INPUT_FILE = Path("evaluation/results/eval_outputs.csv")
OUTPUT_SCORES = Path("evaluation/results/eval_scores.csv")
SUMMARY_FILE = Path("evaluation/results/summary.txt")


# ============================
#  Helper scoring functions
# ============================

def correctness_score(gold, pred):
    """
    For answerable questions:
    score 1 if prediction semantically contains gold answer
    else 0
    """
    if gold is None or gold.strip() == "":
        return 0
    
    gold_low = gold.lower()
    pred_low = pred.lower() if isinstance(pred, str) else ""

    return 1 if gold_low[:20] in pred_low or gold_low in pred_low else 0


def groundedness_score(retrieved_chunks, gold):
    """
    Score 1 if retrieved chunks contain keywords from gold answer.
    """
    if not isinstance(retrieved_chunks, list):
        return 0

    gold_kw = gold.lower().split()
    retrieved_text = " ".join(ch.lower() for ch in retrieved_chunks)

    # simple heuristic: any keyword match → grounded
    for kw in gold_kw[:5]:  # first 5 keywords only
        if kw in retrieved_text:
            return 1
    return 0


def unanswerable_score(system_answer):
    """
    Score safe-refusal:
    If model said: context not enough / cannot answer → 1
    Otherwise → 0
    """
    if not isinstance(system_answer, str):
        return 0
    
    refusal_markers = [
        "not enough information",
        "cannot answer",
        "no sufficient information",
        "context does not provide",
        "not provided in the context",
        "the context doesn't contain"
    ]

    ans = system_answer.lower()
    return 1 if any(m in ans for m in refusal_markers) else 0


# ============================
#  Load evaluation output
# ============================

df = pd.read_csv(INPUT_FILE)

# Convert retrieved_chunks from string → list
def safe_load(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["retrieved_chunks"] = df["retrieved_chunks"].apply(safe_load)

# ============================
#  Apply Scoring
# ============================

scores = []

for idx, row in df.iterrows():

    qid = row["id"]
    is_unans = int(row["is_unanswerable"])
    gold = str(row["gold_answer"])
    pred = str(row["system_answer"])
    retrieved = row["retrieved_chunks"]

    if is_unans == 1:
        c = None
        g = None
        s = unanswerable_score(pred)
    else:
        c = correctness_score(gold, pred)
        g = groundedness_score(retrieved, gold)
        s = None

    scores.append({
        "id": qid,
        "correctness": c,
        "groundedness": g,
        "safe_refusal": s,
        "group": row["group"],
        "mode": row["mode"]
    })

score_df = pd.DataFrame(scores)
score_df.to_csv(OUTPUT_SCORES, index=False)

# ============================
#  Summary
# ============================

summary = []

def avg(col):
    valid = score_df[col].dropna()
    return valid.mean() if len(valid) > 0 else 0

summary.append(f"Correctness avg: {avg('correctness'):.3f}")
summary.append(f"Groundedness avg: {avg('groundedness'):.3f}")
summary.append(f"Safe refusal avg: {avg('safe_refusal'):.3f}")

# group-level summary
group_stats = score_df.groupby("group")[["correctness", "groundedness", "safe_refusal"]].mean()

summary.append("\nGroup-level stats:\n")
summary.append(group_stats.to_string())

SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")

print("=== Scoring completed ===")
print(f"Saved per-question scores → {OUTPUT_SCORES}")
print(f"Saved summary → {SUMMARY_FILE}")
