import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("results/figures").mkdir(parents=True, exist_ok=True)

judge = pd.read_csv("results/judge_score_result.csv")
pairwise = pd.read_csv("results/pairwise_result.csv")
elo = pd.read_csv("results/elo_result.csv")

avg_scores = {
    "high": judge["high_score"].mean(),
    "mid": judge["mid_score"].mean(),
    "low": judge["low_score"].mean(),
}

plt.figure()
plt.bar(avg_scores.keys(), avg_scores.values())
plt.title("Average Judge Scores")
plt.xlabel("Answer Level")
plt.ylabel("Average Score")
plt.savefig("results/figures/average_judge_scores.png")
plt.close()

pairwise_success = pairwise.groupby("expected_winner")["success"].mean()

plt.figure()
plt.bar(pairwise_success.index, pairwise_success.values)
plt.title("Pairwise Success Rate")
plt.xlabel("Expected Winner")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.savefig("results/figures/pairwise_success_rate.png")
plt.close()

plt.figure()
plt.bar(elo["answer_level"], elo["elo_score"])
plt.title("Elo Scores")
plt.xlabel("Answer Level")
plt.ylabel("Elo Score")
plt.savefig("results/figures/elo_scores.png")
plt.close()

print("Saved figures to results/figures/")
