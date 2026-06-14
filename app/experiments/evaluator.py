# evaluator.py
# Judge LLM 호출 + 결과 파싱 + 가중치 점수 계산 (Claude 버전)

import json
import os
from dotenv import load_dotenv
import anthropic

from prompt import build_judge_prompt
from rubric import calculate_weighted_score, normalize_score

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_judge(question: str, category: str, user_answer: str, competitor_answer: str) -> dict:
    """Judge LLM 호출 → raw JSON 결과 반환"""
    prompt = build_judge_prompt(question, category, user_answer, competitor_answer)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def parse_result(raw_result: dict, category: str) -> dict:
    """
    raw Judge 결과 → 가중치 적용 총점 + 정규화 점수 계산
    """
    result = {}

    for key in ["answer_a", "answer_b"]:
        ans = raw_result[key]
        scores = {
            "논리성": ans["논리성"],
            "구체성": ans["구체성"],
            "직무적합성": ans["직무적합성"],
            "완성도": ans["완성도"],
        }
        weighted_total = calculate_weighted_score(scores, category)
        normalized = normalize_score(weighted_total, category)

        result[key] = {
            **scores,
            "weighted_total": weighted_total,
            "normalized_score": normalized,
            "피드백": ans["피드백"],
        }

    result["winner"] = raw_result["winner"]
    result["reason"] = raw_result["reason"]

    return result


def evaluate(question: str, category: str, user_answer: str, competitor_answer: str) -> dict:
    """전체 평가 파이프라인: 호출 → 파싱 → 점수 계산"""
    raw = call_judge(question, category, user_answer, competitor_answer)
    return parse_result(raw, category)


# ── 테스트용 ──────────────────────────────────────────────
if __name__ == "__main__":
    question = "Transformer 아키텍처의 핵심 원리를 설명해보세요."
    category = "AI/NLP 기초"

    user_answer = """
    Transformer는 Self-Attention 메커니즘을 핵심으로 합니다.
    입력 시퀀스의 모든 토큰 쌍 관계를 병렬로 계산하고,
    Positional Encoding으로 순서 정보를 보완합니다.
    RNN과 달리 병렬 연산이 가능해 긴 시퀀스에서도 효과적입니다.
    """

    competitor_answer = """
    Transformer는 Self-Attention이라는 방식으로 단어 간 관계를 파악합니다.
    Encoder와 Decoder로 나뉘며 번역 같은 작업에 사용됩니다.
    기존 RNN보다 병렬 처리가 빠릅니다.
    """

    result = evaluate(question, category, user_answer, competitor_answer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
