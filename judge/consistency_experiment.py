# consistency_experiment.py
# Judge 모델 평가 일관성 실험
# 동일한 질문/답변 쌍을 N회 반복 평가 → 일관성 측정

import json
import statistics
from evaluator import evaluate

# ── 실험 설정 ──────────────────────────────────────────────
N_TRIALS = 5  # 반복 횟수 (비용 고려해서 5회)

TEST_CASES = [
    {
        "question": "Transformer 아키텍처의 핵심 원리를 설명해보세요.",
        "category": "AI/NLP 기초",
        "user_answer": """
        Transformer는 Self-Attention 메커니즘을 핵심으로 합니다.
        입력 시퀀스의 모든 토큰 쌍 관계를 병렬로 계산하고,
        Positional Encoding으로 순서 정보를 보완합니다.
        RNN과 달리 병렬 연산이 가능해 긴 시퀀스에서도 효과적입니다.
        """,
        "competitor_answer": """
        Transformer는 Self-Attention이라는 방식으로 단어 간 관계를 파악합니다.
        Encoder와 Decoder로 나뉘며 번역 같은 작업에 사용됩니다.
        기존 RNN보다 병렬 처리가 빠릅니다.
        """,
    },
    {
        "question": "팀 프로젝트에서 갈등이 생겼을 때 어떻게 해결했나요?",
        "category": "인성",
        "user_answer": """
        NLP 팀 프로젝트에서 데이터 전처리 방식을 두고 의견 충돌이 있었습니다.
        감정적 논쟁 대신 두 방식 모두 실험해보자고 제안하고 F1 Score 기준으로 비교했습니다.
        제 방식이 7% 높게 나왔고, 팀원도 데이터로 납득했습니다.
        """,
        "competitor_answer": """
        프로젝트 중 팀원과 구현 방향에 대해 의견이 달랐는데,
        각자 이유를 이야기하고 절충안을 찾아서 진행했습니다.
        대화를 통해 해결했고 프로젝트는 무사히 마쳤습니다.
        """,
    },
]


def run_consistency_experiment(test_case: dict, n_trials: int) -> dict:
    """
    단일 테스트 케이스에 대해 N회 반복 평가 → 일관성 지표 계산
    """
    scores_user = []
    scores_competitor = []
    winners = []

    print(f"\n[실험] {test_case['question'][:30]}...")
    for i in range(n_trials):
        print(f"  시도 {i+1}/{n_trials}...", end=" ")
        result = evaluate(
            test_case["question"],
            test_case["category"],
            test_case["user_answer"],
            test_case["competitor_answer"],
        )
        scores_user.append(result["answer_a"]["normalized_score"])
        scores_competitor.append(result["answer_b"]["normalized_score"])
        winners.append(result["winner"])
        print(f"user={result['answer_a']['normalized_score']}, winner={result['winner']}")

    # 일관성 지표 계산
    winner_consistency = winners.count(winners[0]) / n_trials * 100  # 첫 번째 결과와 일치율

    return {
        "question": test_case["question"],
        "category": test_case["category"],
        "n_trials": n_trials,
        "user_scores": {
            "values": scores_user,
            "mean": round(statistics.mean(scores_user), 1),
            "stdev": round(statistics.stdev(scores_user), 2) if n_trials > 1 else 0,
            "min": min(scores_user),
            "max": max(scores_user),
        },
        "competitor_scores": {
            "values": scores_competitor,
            "mean": round(statistics.mean(scores_competitor), 1),
            "stdev": round(statistics.stdev(scores_competitor), 2) if n_trials > 1 else 0,
            "min": min(scores_competitor),
            "max": max(scores_competitor),
        },
        "winners": winners,
        "winner_consistency_pct": round(winner_consistency, 1),
        "verdict": "일관성 높음 ✅" if winner_consistency >= 80 else "일관성 낮음 ⚠️",
    }


def run_all_experiments():
    """전체 테스트 케이스 실험 실행 + 최종 리포트"""
    all_results = []

    for test_case in TEST_CASES:
        result = run_consistency_experiment(test_case, N_TRIALS)
        all_results.append(result)

    # 최종 리포트 출력
    print("\n" + "="*60)
    print("Judge 모델 평가 일관성 실험 결과")
    print("="*60)

    for r in all_results:
        print(f"\n📌 질문: {r['question'][:40]}...")
        print(f"   카테고리: {r['category']}")
        print(f"   사용자 점수: 평균 {r['user_scores']['mean']}점 (±{r['user_scores']['stdev']})")
        print(f"   경쟁자 점수: 평균 {r['competitor_scores']['mean']}점 (±{r['competitor_scores']['stdev']})")
        print(f"   승자 판정: {r['winners']}")
        print(f"   승자 일치율: {r['winner_consistency_pct']}% → {r['verdict']}")

    # JSON 저장
    with open("consistency_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n✅ 결과 저장 완료: consistency_results.json")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
