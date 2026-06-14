import pandas as pd
from pathlib import Path

PAIRWISE_PATH = "results/pairwise_result.csv"
OUT_PATH = "results/elo_result.csv"

Path("results").mkdir(exist_ok=True)

df = pd.read_csv(PAIRWISE_PATH)

ratings = {
    "high": 1000.0,
    "mid": 1000.0,
    "low": 1000.0,
}

K = 32

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

for _, row in df.iterrows():
    a = row["answer_a"]
    b = row["answer_b"]
    winner = row["winner"]

    r_a = ratings[a]
    r_b = ratings[b]

    e_a = expected_score(r_a, r_b)
    e_b = expected_score(r_b, r_a)

    if winner == a:
        s_a, s_b = 1, 0
    elif winner == b:
        s_a, s_b = 0, 1
    else:
        s_a, s_b = 0.5, 0.5

    ratings[a] = r_a + K * (s_a - e_a)
    ratings[b] = r_b + K * (s_b - e_b)

out = pd.DataFrame([
    {"answer_level": level, "elo_score": round(score, 2)}
    for level, score in ratings.items()
])

out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT_PATH}")
print(out)
print("Elo success:", ratings["high"] > ratings["mid"] > ratings["low"])
