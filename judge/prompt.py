# prompt.py
# Judge LLM에게 전달할 프롬프트 템플릿

from rubric import RUBRICS, SCORE_CRITERIA


def build_judge_prompt(question: str, category: str, user_answer: str, competitor_answer: str) -> str:
    rubric_info = RUBRICS[category]
    criteria_text = "\n".join(
        f"- {item} (1~5점): {desc}"
        for item, desc in SCORE_CRITERIA.items()
    )
    weight_text = "\n".join(
        f"- {item}: ×{w}"
        for item, w in rubric_info["weights"].items()
    )

    prompt = f"""당신은 AI/NLP 분야 전문 면접 평가관입니다.
지원자와 AI 경쟁자의 면접 답변을 객관적으로 비교 평가하세요.

[면접 질문]
{question}

[카테고리]
{category}

[좋은 답변 기준]
{rubric_info["description"]}

[평가 항목 및 기준]
{criteria_text}

[카테고리별 가중치]
{weight_text}

[답변 A] (지원자)
{user_answer}

[답변 B] (AI 경쟁자)
{competitor_answer}

---

위 두 답변을 각 평가 항목별로 1~5점으로 채점하세요.
편향 없이 답변 내용만을 기준으로 평가하세요.
답변 순서(A/B)에 의한 편향이 생기지 않도록 주의하세요.

반드시 아래 JSON 형식으로만 응답하세요. JSON 외 텍스트는 절대 포함하지 마세요.

{{
  "answer_a": {{
    "논리성": 점수,
    "구체성": 점수,
    "직무적합성": 점수,
    "완성도": 점수,
    "피드백": "답변 A의 강점과 약점을 1~2문장으로"
  }},
  "answer_b": {{
    "논리성": 점수,
    "구체성": 점수,
    "직무적합성": 점수,
    "완성도": 점수,
    "피드백": "답변 B의 강점과 약점을 1~2문장으로"
  }},
  "winner": "A" 또는 "B" 또는 "TIE",
  "reason": "승자 판정 이유를 2~3문장으로"
}}"""

    return prompt
