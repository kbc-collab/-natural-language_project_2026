# rubric.py
# 카테고리별 평가 기준(rubric)과 항목별 가중치 정의

RUBRICS = {
    "인성": {
        "description": "구체적 상황 제시, 본인의 역할 명확화, 갈등/성장 과정 논리적 서술, 결과와 배운 점 포함",
        "weights": {
            "논리성": 2,
            "구체성": 2,
            "직무적합성": 1,
            "완성도": 1,
        },
    },
    "AI/NLP 기초": {
        "description": "핵심 개념 정확히 설명, 기술 간 차이점 이해, 실제 적용 사례 언급",
        "weights": {
            "논리성": 1,
            "구체성": 1,
            "직무적합성": 3,
            "완성도": 1,
        },
    },
    "프로젝트 경험": {
        "description": "문제를 구체적으로 정의, 접근 방법 논리적 서술, 결과 수치화, 재현 가능한 해결책 제시",
        "weights": {
            "논리성": 2,
            "구체성": 3,
            "직무적합성": 2,
            "완성도": 1,
        },
    },
    "문제해결/실무": {
        "description": "요구사항 분석 우선, 복수의 접근법 비교, 선택 이유 명확, 실무 경험 기반",
        "weights": {
            "논리성": 2,
            "구체성": 2,
            "직무적합성": 3,
            "완성도": 1,
        },
    },
}

SCORE_CRITERIA = {
    "논리성": "답변 구조가 명확하고 일관성이 있는가. 두괄식 혹은 STAR 구조로 전달되는가.",
    "구체성": "수치, 사례, 실제 경험이 구체적으로 포함됐는가. 추상적 표현 없이 근거가 있는가.",
    "직무적합성": "AI/NLP 직무와 연결되는가. 기술 용어와 개념을 적절히 사용했는가.",
    "완성도": "질문의 의도를 파악하고 충분히 답했는가. 핵심을 빠뜨리지 않았는가.",
}


def get_max_score(category: str) -> int:
    weights = RUBRICS[category]["weights"]
    return sum(weights.values()) * 5  # 각 항목 최대 5점


def normalize_score(raw_score: float, category: str) -> float:
    """raw_score를 100점 만점으로 정규화"""
    max_score = get_max_score(category)
    return round((raw_score / max_score) * 100, 1)


def calculate_weighted_score(scores: dict, category: str) -> float:
    """항목별 점수 + 가중치 적용해 총점 계산"""
    weights = RUBRICS[category]["weights"]
    total = sum(scores[item] * weights[item] for item in weights)
    return total
