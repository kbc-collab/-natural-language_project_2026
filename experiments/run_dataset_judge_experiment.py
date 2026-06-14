import pandas as pd
from pathlib import Path

DATA_PATH = "dataset_step2.csv"
OUT_PATH = "results/judge_score_result.csv"

Path("results").mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

def simple_judge_score(question, criteria, answer):
    if pd.isna(answer):
        return 0

    question = str(question)
    criteria = str(criteria)
    answer = str(answer)

    score = 0

    # 답변 길이
    score += min(len(answer) / 250, 3)

    # 좋은 답변 기준과 겹치는 키워드 반영
    criteria_words = set(criteria.replace(",", " ").replace(".", " ").split())
    answer_words = set(answer.replace(",", " ").replace(".", " ").split())
    overlap = len(criteria_words & answer_words)

    score += min(overlap * 0.4, 4)

    # 구조적 답변 가산점
    structure_keywords = ["첫째", "둘째", "따라서", "예를", "경험", "문제", "해결", "근거", "배웠습니다"]
    score += sum(1 for k in structure_keywords if k in answer) * 0.4

    return round(min(score, 10), 2)

results = []
for _, row in df.iterrows():
    question_id = row["#"]
    category = row["카테고리"]
    difficulty = row["난이도"]
    question = row["질문"]
    criteria = row["좋은 답변 기준"]

    high_score = simple_judge_score(question, criteria, row["상 답변"])
    mid_score = simple_judge_score(question, criteria, row["중 답변"])
    low_score = simple_judge_score(question, criteria, row["하 답변"])

    success = high_score > mid_score > low_score

    results.append({
        "question_id": question_id,
        "category": category,
        "difficulty": difficulty,
        "high_score": high_score,
        "mid_score": mid_score,
        "low_score": low_score,
        "success": success
    })

out = pd.DataFrame(results)
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT_PATH}")
print(out.head())
print("Success rate:", out["success"].mean())
