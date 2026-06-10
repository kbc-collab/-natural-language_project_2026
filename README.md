# -natural-language_project_2026

# Judge 평가 모듈

- ** 평가 모델 및 Scoring 설계
- ** LLM-as-a-Judge 기반 답변 평가 + Elo Rating 순위 산정

---

## 파일 구조

```
judge/
├── rubric.py                  # 카테고리별 평가 기준 + 가중치
├── prompt.py                  # Judge 프롬프트 템플릿
├── evaluator.py               # Claude API 호출 + 점수 계산
├── elo.py                     # Elo Rating 알고리즘
├── pipeline.py                # 전체 파이프라인 연결
├── judge_agent.py             # 스키마 연결 인터페이스
└── consistency_experiment.py  # Judge 일관성 실험
```

---

## 핵심 기술

### 1. LLM-as-a-Judge
Claude API를 Judge로 활용해 사용자 답변과 AI 경쟁자 답변을 비교 평가한다.
카테고리(인성 / AI·NLP 기초 / 프로젝트 경험 / 문제해결·실무)별로 평가 기준(rubric)과 항목별 가중치를 다르게 적용한다.

**평가 항목 (5개)**
| 항목 | 설명 |
|---|---|
| 논리성 | 답변 구조가 명확하고 일관성이 있는가 |
| 구체성 | 수치, 사례, 경험이 구체적으로 포함됐는가 |
| 직무적합성 | AI/NLP 직무와 연결되는가 |
| 완성도 | 질문의 의도를 파악하고 충분히 답했는가 |
| 전문성 | 직무 관련 전문 지식을 정확히 사용했는가 |

### 2. Pairwise Comparison
사용자 답변(A)과 AI 경쟁자 답변(B)을 1:1 비교해 승자를 판정한다.
Judge는 `A_WINS` / `B_WINS` / `TIE` 중 하나를 반환하며 판정 근거도 함께 제공한다.

### 3. Elo Rating
체스의 Elo Rating 알고리즘을 면접 평가에 적용해 상대적 순위를 산정한다.

```
초기 점수: 1000점
K Factor: 32
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R_A_new = R_A + K * (S_A - E_A)
```

사용자는 AI 경쟁자(상/중/하) 3종과 매칭되며, 라운드가 쌓일수록 실력에 맞는 Elo 점수로 수렴한다.

---

## 실험 결과 — Judge 일관성 검증

동일한 질문/답변 쌍을 5회 반복 평가해 Judge 모델의 신뢰도를 검증했다.

| 질문 | 사용자 점수 평균 | 표준편차 | 승자 일치율 |
|---|---|---|---|
| Transformer 핵심 원리 | 93.3점 | ±0.0 | 100% |
| 팀 갈등 해결 경험 | 90.0점 | ±0.0 | 100% |

> 승자 일치율 100%, 점수 표준편차 0.0으로 높은 평가 일관성 확인

---

## 인터페이스 연결

`judge_agent.py`는 팀원 A의 `schemas/models.py` 스키마를 따른다.

```python
from judge.judge_agent import evaluate

result = evaluate(evaluation_request)  # EvaluationRequest → EvaluationResult
```

**입력:** `EvaluationRequest` (question, answers 목록)
**출력:** `EvaluationResult` (점수, Elo, 순위, 피드백, 개선 제안)

---

## 환경 설정

```bash
pip install anthropic python-dotenv
```

`.env` 파일 생성:
```
ANTHROPIC_API_KEY=your_api_key
```

---

## 실행 방법

```bash
# Judge 단일 평가 테스트
python evaluator.py

# 전체 파이프라인 테스트
python pipeline.py

# llm 에이전트 인터페이스 테스트
python judge_agent.py

# 일관성 실험
python consistency_experiment.py
```
