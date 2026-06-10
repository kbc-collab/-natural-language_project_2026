# judge_agent.py
# 팀원 A의 schemas/models.py 인터페이스에 맞는 Judge Agent 구현
# EvaluationRequest → EvaluationResult

import os
import json
from dotenv import load_dotenv
import anthropic

from rubric import RUBRICS, SCORE_CRITERIA

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── 팀원 A 스키마에 맞춘 Elo 설정 ──────────────────────────
INITIAL_ELO = 1000.0
K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a: float, rating_b: float, result: str):
    """result: 'A_WINS' | 'B_WINS' | 'TIE'"""
    ea = expected_score(rating_a, rating_b)
    eb = expected_score(rating_b, rating_a)
    if result == "A_WINS":
        sa, sb = 1.0, 0.0
    elif result == "B_WINS":
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5
    return round(rating_a + K_FACTOR * (sa - ea), 1), round(rating_b + K_FACTOR * (sb - eb), 1)


def build_prompt(request: dict) -> str:
    """EvaluationRequest dict → Judge 프롬프트 생성"""
    question = request["question_content"]
    intent = request["question_intent"]
    criteria = request.get("evaluation_criteria") or "논리성, 전문성, 직무적합성, 명확성, 구체성 기준으로 평가"
    target_job = request["target_job"]
    answers = request["answers"]

    answers_text = ""
    for i, ans in enumerate(answers):
        label = "사용자" if ans["respondent_type"] == "USER" else f"AI 경쟁자 (레벨: {ans.get('competitor_level', '미지정')})"
        answers_text += f"\n[답변 {'A' if i == 0 else 'B'}] {label}\n{ans['content']}\n"

    prompt = f"""당신은 {target_job} 면접 전문 평가관입니다.

[면접 질문]
{question}

[질문 의도]
{intent}

[평가 기준]
{criteria}

[답변 목록]
{answers_text}

각 답변을 아래 5개 항목으로 0~100점 채점하고, Pairwise 비교를 수행하세요.

평가 항목:
- logic (논리성): 답변 구조가 명확하고 일관성이 있는가
- professionalism (전문성): 직무 관련 전문 지식을 정확히 사용했는가
- relevance (직무적합성): 목표 직무({target_job})와 연결되는가
- clarity (명확성): 핵심을 명확하게 전달했는가
- concreteness (구체성): 수치, 사례, 경험이 구체적으로 포함됐는가

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "answer_a": {{
    "logic": 점수,
    "professionalism": 점수,
    "relevance": 점수,
    "clarity": 점수,
    "concreteness": 점수,
    "feedback": "답변 A 피드백"
  }},
  "answer_b": {{
    "logic": 점수,
    "professionalism": 점수,
    "relevance": 점수,
    "clarity": 점수,
    "concreteness": 점수,
    "feedback": "답변 B 피드백"
  }},
  "pairwise_result": "A_WINS" 또는 "B_WINS" 또는 "TIE",
  "pairwise_reasoning": "비교 판단 근거 2~3문장",
  "comparative_feedback": "두 답변 비교 분석",
  "improvement_tip": "사용자를 위한 구체적 개선 제안"
}}"""
    return prompt


def evaluate(request: dict) -> dict:
    """
    팀원 A 인터페이스 메서드
    EvaluationRequest(dict) → EvaluationResult(dict)
    """
    prompt = build_prompt(request)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    judge_result = json.loads(raw.strip())

    # Elo 계산
    elo_a, elo_b = update_elo(INITIAL_ELO, INITIAL_ELO, judge_result["pairwise_result"])

    # total_score 계산 (5개 항목 평균)
    def total(scores): return round(sum(scores.values()) / len(scores), 1)

    scores_a = {k: judge_result["answer_a"][k] for k in ["logic", "professionalism", "relevance", "clarity", "concreteness"]}
    scores_b = {k: judge_result["answer_b"][k] for k in ["logic", "professionalism", "relevance", "clarity", "concreteness"]}

    # rank 결정
    rank_a = 1 if judge_result["pairwise_result"] in ["A_WINS", "TIE"] else 2
    rank_b = 1 if judge_result["pairwise_result"] == "B_WINS" else (1 if judge_result["pairwise_result"] == "TIE" else 2)

    answers = request["answers"]

    result = {
        "question_id": request["question_id"],
        "evaluations": [
            {
                "respondent_type": answers[0]["respondent_type"],
                "competitor_id": answers[0].get("competitor_id"),
                "score_breakdown": scores_a,
                "total_score": total(scores_a),
                "elo_score": elo_a,
                "feedback": judge_result["answer_a"]["feedback"],
                "rank": rank_a,
            },
            {
                "respondent_type": answers[1]["respondent_type"],
                "competitor_id": answers[1].get("competitor_id"),
                "score_breakdown": scores_b,
                "total_score": total(scores_b),
                "elo_score": elo_b,
                "feedback": judge_result["answer_b"]["feedback"],
                "rank": rank_b,
            },
        ],
        "pairwise_comparisons": [
            {
                "answer_a_respondent": answers[0]["respondent_type"],
                "answer_b_respondent": answers[1]["respondent_type"],
                "result": judge_result["pairwise_result"],
                "reasoning": judge_result["pairwise_reasoning"],
            }
        ],
        "elo_config": {
            "initial_score": INITIAL_ELO,
            "k_factor": K_FACTOR,
        },
        "comparative_feedback": judge_result["comparative_feedback"],
        "improvement_tip": judge_result["improvement_tip"],
    }

    return result


# ── 테스트용 ──────────────────────────────────────────────
if __name__ == "__main__":
    request = {
        "question_id": "q001",
        "question_content": "Transformer 아키텍처의 핵심 원리를 설명해보세요.",
        "question_intent": "Transformer 핵심 개념 이해도 평가",
        "evaluation_criteria": "Self-Attention 메커니즘 이해, Positional Encoding 역할, RNN과의 차별점",
        "target_job": "AI/NLP 개발자",
        "answers": [
            {
                "respondent_type": "USER",
                "content": "Transformer는 Self-Attention 메커니즘을 핵심으로 합니다. 입력 시퀀스의 모든 토큰 쌍 관계를 병렬로 계산하고, Positional Encoding으로 순서 정보를 보완합니다.",
            },
            {
                "respondent_type": "AI_COMPETITOR",
                "competitor_id": 2,
                "competitor_level": "mid",
                "content": "Transformer는 Self-Attention이라는 방식으로 단어 간 관계를 파악합니다. Encoder와 Decoder로 나뉘며 번역 같은 작업에 사용됩니다.",
            },
        ],
    }

    result = evaluate(request)
    print(json.dumps(result, ensure_ascii=False, indent=2))
