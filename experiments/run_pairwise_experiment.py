import pandas as pd
from pathlib import Path

DATA_PATH = "results/judge_score_result.csv"
OUT_PATH = "results/pairwise_result.csv"

Path("results").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
results = []

for _, row in df.iterrows():
    qid = row["question_id"]

    pairs = [
        ("high", "mid", row["high_score"], row["mid_score"], "high"),
        ("high", "low", row["high_score"], row["low_score"], "high"),
        ("mid", "low", row["mid_score"], row["low_score"], "mid"),
    ]

    for a_name, b_name, a_score, b_score, expected in pairs:
        if a_score > b_score:
            winner = a_name
        elif b_score > a_score:
            winner = b_name
        else:
            winner = "tie"

        results.append({
            "question_id": qid,
            "answer_a": a_name,
            "answer_b": b_name,
            "score_a": a_score,
            "score_b": b_score,
            "winner": winner,
            "expected_winner": expected,
            "success": winner == expected
        })

out = pd.DataFrame(results)
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT_PATH}")
print(out.head())
print("Pairwise success rate:", out["success"].mean())
