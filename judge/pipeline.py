# pipeline.py
# Judge 평가 + Elo Rating 전체 파이프라인

import json
from evaluator import evaluate
from elo import EloSystem

# 경쟁자 수준 매핑
COMPETITOR_LEVEL_MAP = {
    "상": "high",
    "중": "mid",
    "하": "low",
}


class JudgePipeline:
    def __init__(self):
        self.elo = EloSystem()
        self.results = []  # 전체 라운드 결과 기록

    def run(self, question: str, category: str, competitor_level: str,
            user_answer: str, competitor_answer: str) -> dict:
        """
        한 라운드 실행
        - Judge 평가 → Elo 업데이트 → 결과 반환
        """
        # 1. Judge 평가
        eval_result = evaluate(question, category, user_answer, competitor_answer)

        # 2. Elo 업데이트
        level_key = COMPETITOR_LEVEL_MAP[competitor_level]
        self.elo.record_match(
            competitor_level=level_key,
            winner=eval_result["winner"],
            question=question,
            normalized_score_user=eval_result["answer_a"]["normalized_score"],
            normalized_score_competitor=eval_result["answer_b"]["normalized_score"],
        )

        # 3. 결과 정리
        round_result = {
            "question": question,
            "category": category,
            "competitor_level": competitor_level,
            "scores": {
                "user": eval_result["answer_a"]["normalized_score"],
                "competitor": eval_result["answer_b"]["normalized_score"],
            },
            "winner": eval_result["winner"],
            "reason": eval_result["reason"],
            "feedback": {
                "user": eval_result["answer_a"]["피드백"],
                "competitor": eval_result["answer_b"]["피드백"],
            },
            "elo_after": {
                "user": self.elo.ratings["user"],
                f"competitor_{level_key}": self.elo.ratings[f"competitor_{level_key}"],
            },
        }

        self.results.append(round_result)
        return round_result

    def get_final_report(self) -> dict:
        """전체 라운드 종료 후 최종 리포트 반환"""
        return {
            "total_rounds": len(self.results),
            "final_ratings": self.elo.ratings,
            "rankings": self.elo.get_rankings(),
            "user_level": self.elo.get_user_level(),
            "round_results": self.results,
        }


# ── 테스트용 ──────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = JudgePipeline()

    # 라운드 1
    result1 = pipeline.run(
        question="Transformer 아키텍처의 핵심 원리를 설명해보세요.",
        category="AI/NLP 기초",
        competitor_level="중",
        user_answer="""
        Transformer는 Self-Attention 메커니즘을 핵심으로 합니다.
        입력 시퀀스의 모든 토큰 쌍 관계를 병렬로 계산하고,
        Positional Encoding으로 순서 정보를 보완합니다.
        RNN과 달리 병렬 연산이 가능해 긴 시퀀스에서도 효과적입니다.
        """,
        competitor_answer="""
        Transformer는 Self-Attention이라는 방식으로 단어 간 관계를 파악합니다.
        Encoder와 Decoder로 나뉘며 번역 같은 작업에 사용됩니다.
        기존 RNN보다 병렬 처리가 빠릅니다.
        """,
    )

    print("=== 라운드 1 결과 ===")
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    # 최종 리포트
    print("\n=== 최종 리포트 ===")
    report = pipeline.get_final_report()
    print(f"사용자 Elo: {report['final_ratings']['user']}")
    print(f"사용자 수준: {report['user_level']}")
    print("\n순위:")
    for r in report["rankings"]:
        print(f"  {r['rank']}위: {r['player']} ({r['rating']}점)")
