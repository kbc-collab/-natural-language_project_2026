#  AI Interview Arena
### Multi-Agent 기반 경쟁형 AI 면접 시뮬레이션 시스템

> **자연어처리 과목 팀 프로젝트 | 2조 | 안수빈 · 안은정 · 김병찬**

---

## 프로젝트 개요

AI Interview Arena는 **Multi-Agent 아키텍처**를 기반으로 실제 취업 경쟁 환경을 시뮬레이션하는 AI 면접 플랫폼입니다.

기존 AI 면접 도구들은 단순히 사용자의 답변에 피드백을 주는 데 그쳤습니다. 이 프로젝트는 **AI 경쟁자와 실시간으로 비교 평가받는 경쟁 구조**를 도입하여, 사용자가 자신의 상대적 위치를 객관적으로 파악할 수 있도록 설계했습니다.

### 핵심 차별점

| 기존 AI 면접 도구 | AI Interview Arena |
|---|---|
| 절대 점수만 제공 | 경쟁자 대비 **상대적 위치** 제공 |
| 단일 AI 모델 사용 | **3개 Agent의 역할 분리** |
| 정적 피드백 | **Elo Rating** 기반 동적 순위 산정 |
| 경쟁 상황 미반영 | 실제 취업 경쟁 환경 시뮬레이션 |

---

## 프로젝트 배경 및 필요성

### AI 윤리적 고려

이 프로젝트는 강의에서 다룬 **AI 윤리 원칙**을 설계 단계부터 반영하였습니다.

**1. 공정성 (Fairness)**
- AI 경쟁자 답변을 수준별(상/중/하)로 분리 생성하여 **편향 없는 비교 기준** 제공
- 아마존 AI 채용 시스템의 성별 편향 사례를 반면교사로, 평가 기준에 인구통계학적 요소 완전 배제
- 카테고리(인성/기술/HR)와 직무별 맞춤 평가 루브릭 적용

**2. 투명성 (Transparency)**
- `LLM-as-a-Judge`의 평가 근거를 `reasoning` 필드로 명시적 반환
- Pairwise 비교 결과와 Elo 점수 계산 과정을 사용자에게 단계별로 공개
- **XAI(설명 가능한 AI)** 원칙을 따라 "왜 이 순위인가"를 항목별 점수 분해(ScoreBreakdown)로 제시

**3. 책임성 (Accountability)**
- **Human-in-the-loop** 구조 유지: 모든 최종 평가는 사용자가 검토하고 수용 여부를 판단
- Judge Agent를 Stub 구조로 설계하여 팀 간 구현 책임을 명확히 분리
- 할루시네이션 방지: `dataset_step2.csv`의 검증된 데이터를 평가 기준 루브릭으로 활용

**4. 프라이버시 (Privacy)**
- 이루다 개인정보 유출 사례를 참고하여, 사용자 답변 데이터를 서버에 저장하지 않는 세션 기반 설계
- `.env.example`로 API 키 등 민감 정보를 코드에서 완전 분리

---

## 🛠️ 시스템 구조

### Multi-Agent 아키텍처

```
사용자 답변 입력
      ↓
┌─────────────────────────────────────────────────────┐
│                  LangGraph State Machine            │
│                 (interview_service.py)              │
│                                                     │
│  [Interviewer Agent] → 직무/유형별 질문 생성         │
│         ↓                                           │
│  [Competitor Agent Pool] → 수준별 답변 병렬 생성     │
│    ├── Level 1 (LOW):  온도 0.9, 짧고 추상적         │
│    ├── Level 2 (MID):  온도 0.7, STAR 시도           │
│    └── Level 3 (HIGH): 온도 0.4, 완벽한 구조+수치    │
│         ↓                                           │
│  [Judge Agent] → LLM-as-a-Judge 평가               │
│    ├── Pairwise Comparison (N*(N-1)/2 쌍)           │
│    └── Elo Rating 산정                              │
│         ↓                                           │
│  순위 & 피드백 출력                                  │
└─────────────────────────────────────────────────────┘
```

### Agent별 역할

| Agent | 역할 | 구현 기술 |
|---|---|---|
| **Interviewer Agent** | 직무·면접 유형 맞춤 질문 생성, 꼬리질문, 중복 방지 | LangChain + 데이터셋 few-shot |
| **Competitor Agent** | 수준별(상/중/하) AI 경쟁자 답변 병렬 생성 | asyncio.gather + LangChain |
| **Judge Agent** | 모든 답변 비교 평가 + Elo 점수 산정 | LLM-as-a-Judge + Pairwise |

---

## 📁 프로젝트 구조

```
📦 AI-Interview-Arena
├── streamlit_app.py              # 메인 UI (Streamlit)
├── dataset_step2.csv             # 면접 질문 및 수준별 답변 데이터셋
├── requirements.txt
├── .env.example                  # 환경변수 템플릿
│
├── app/
│   ├── agents/
│   │   ├── interviewer_agent.py  # 면접관 Agent
│   │   ├── competitor_agent.py   # 경쟁자 Agent (수준별 3종)
│   │   └── judge_agent.py        # 평가 Judge Agent (Stub → 교체 가능)
│   │
│   ├── core/
│   │   └── prompts/              # 수준별 프롬프트 템플릿
│   │       ├── interviewer.txt
│   │       ├── competitor_level1.txt  # LOW (초보)
│   │       ├── competitor_level2.txt  # MID (보통)
│   │       └── competitor_level3.txt  # HIGH (우수)
│   │
│   ├── experiments/
│   │   ├── elo.py                # Elo Rating 알고리즘 구현
│   │   └── evaluator.py          # Claude API 연동 평가 파이프라인
│   │
│   ├── services/
│   │   └── interview_service.py  # LangGraph 워크플로우 (State Machine)
│   │
│   ├── schemas/
│   │   └── models.py             # Pydantic 데이터 스키마 (팀 간 계약)
│   │
│   └── api/
│       └── interview.py          # FastAPI 엔드포인트
│
└── tests/
    └── test_agents.py
```

---

## 핵심 기술 설명

### 1. LLM-as-a-Judge

단순 점수 부여가 아닌, **강력한 LLM이 다수의 답변을 동시에 읽고 비교 평가**하는 방식입니다.

```python
# app/agents/judge_agent.py
class JudgeAgentStub:
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        # 5개 항목 세부 점수 산출
        scores = ScoreBreakdown(
            logic=...,           # 논리성
            professionalism=..., # 전문성
            relevance=...,       # 직무 적합성
            clarity=...,         # 명확성
            concreteness=...,    # 구체성 (수치/사례)
        )
```

평가 항목 5가지는 발표자료 및 강의에서 강조한 **설명 가능한 AI** 원칙에 따라, 각 항목이 왜 중요한지 사용자가 이해할 수 있도록 설계했습니다.

### 2. Pairwise Comparison + Elo Rating

```
답변 A vs 답변 B → A_WINS / B_WINS / TIE
답변 A vs 답변 C → ...
...
(N*(N-1)/2 쌍 비교)
         ↓
Elo Rating 업데이트:
  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
  R_A_new = R_A + K * (S_A - E_A)
         ↓
최종 순위 산정
```

체스의 Elo 시스템을 면접 평가에 적용하여, **절대 점수가 아닌 상대적 강도**를 측정합니다.

```python
# app/experiments/elo.py
class EloSystem:
    def get_user_level(self) -> str:
        """사용자 Elo 점수와 경쟁자 점수를 비교해 상대적 위치 반환"""
        if user_rating >= high:   return "상위권 (AI 고수준 경쟁자 초과)"
        elif user_rating >= mid:  return "중상위권"
        elif user_rating >= low:  return "중위권"
        else:                     return "하위권 (AI 저수준 경쟁자 미달)"
```

### 3. LangGraph State Machine

면접 전체 흐름을 **상태 기반 그래프**로 관리하여, 각 단계의 실패 시 fallback 처리와 루프 제어가 가능합니다.

```
START
  → generate_question        (Interviewer Agent)
  → generate_competitor_answers  (Competitor Pool, 병렬)
  → evaluate_answers         (Judge Agent)
  → [check_continue]
       ├─ 질문 수 미달: generate_question 루프
       └─ 5문항 완료: generate_report → END
```

### 4. 수준별 Competitor Prompt Engineering

```
LOW  (temperature=0.9):  열정은 있으나 준비 부족, 추상적 답변
MID  (temperature=0.7):  STAR 기법 시도하나 수치 부족
HIGH (temperature=0.4):  완벽한 STAR + 구체적 수치 + 전문 용어
```

온도 파라미터를 수준별로 차별화하여, 낮은 수준 경쟁자는 더 다양하고 불규칙한 답변을, 높은 수준 경쟁자는 일관되고 전문적인 답변을 생성합니다.

---

## 데이터셋

`dataset_step2.csv`는 면접 유형(인성/기술/HR)별로 구성되며 다음 컬럼을 포함합니다.

| 컬럼 | 설명 |
|---|---|
| `카테고리` | 면접 유형 (인성면접 / 기술면접 / HR면접) |
| `난이도` | 질문 난이도 |
| `질문` | 면접 질문 본문 |
| `하 답변` | Level 1 경쟁자 예시 답변 |
| `중 답변` | Level 2 경쟁자 예시 답변 |
| `상 답변` | Level 3 경쟁자 예시 답변 |
| `좋은 답변 기준` | Judge Agent 평가 루브릭 |

데이터셋은 Competitor Agent의 **few-shot 학습 예시**로, Interviewer Agent의 **질문 시드**로, Judge Agent의 **평가 기준(루브릭)**으로 세 가지 용도로 활용됩니다.

---

## 실행 방법

### 1. 환경 설정

```bash
git clone https://github.com/kbc-collab/-natural-language_project_2026.git
cd -natural-language_project_2026
pip install -r requirements.txt
```

```bash
# .env 파일 생성
cp .env.example .env
# .env 파일에 API 키 입력
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here   # 선택 사항
```

### 2. Streamlit UI 실행

```bash
streamlit run streamlit_app.py
```

### 3. 면접 진행 흐름

```
① 사이드바에서 카테고리 / 난이도 선택
② 면접 질문 선택 (또는 Interviewer Agent 자동 생성)
③ 텍스트 영역에 답변 입력
④ "AI 경쟁 면접 평가 시작" 클릭
⑤ AI 경쟁자 3명의 답변 확인
⑥ Judge Agent의 Pairwise 비교 평가 결과 및 Elo 순위 확인
⑦ 항목별 세부 점수(Radar Chart) 및 개선 피드백 확인
```

---

## 팀 역할 분담 및 모듈 계약

이 프로젝트는 **인터페이스 계약(Interface Contract)** 방식으로 팀 간 독립 개발을 진행했습니다. `app/schemas/models.py`가 팀 간의 단일 계약 파일 역할을 합니다.

| 담당 | 모듈 | 핵심 계약 |
|---|---|---|
| **A (LLM Agent 설계)** | `agents/`, `services/`, `core/prompts/` | `EvaluationRequest` 입력 형식 정의 |
| **B (평가모델 & Scoring)** | `agents/judge_agent.py`, `experiments/elo.py` | `EvaluationResult` 출력 형식 준수 |
| **C (데이터셋 & 통합)** | `experiments/dataset_loader.py`, `streamlit_app.py` | `QuestionContext` 형식으로 RAG 데이터 제공 |

Judge Agent는 `JudgeAgentStub`(Mock)으로 먼저 인터페이스를 확정하고, 평가팀이 실제 LLM 로직으로 교체하는 구조입니다. 교체 시 코드 한 줄만 수정하면 됩니다.

```python
# judge_agent.py 하단 — 교체 시 이 한 줄만 변경
# from app.agents.judge_agent_impl import JudgeAgent as JudgeAgentStub
judge_agent = JudgeAgentStub()
```

---

## 실험 설계

### Prompt 비교 실험
- 동일 질문에 대해 3가지 페르소나 프롬프트로 답변 생성
- Judge Agent의 평가 일관성 측정 (같은 입력 → 유사한 순위 산출 여부)

### 경쟁자 수준별 변별력 확인
- LOW / MID / HIGH 경쟁자 답변이 실제로 다른 점수를 받는지 검증
- Elo 점수 분포가 수준에 따라 유의미하게 분리되는지 확인

### Judge 모델 신뢰도 검증
- 동일 평가를 복수 실행하여 결과 일관성(재현성) 측정
- Pairwise 결과의 이행성 확인 (A>B, B>C이면 A>C여야 함)

---

## 사용 기술 스택

| 분류 | 기술 |
|---|---|
| **LLM Framework** | LangChain 0.3, LangGraph 0.4 |
| **LLM 모델** | Claude (Anthropic), GPT-4o (OpenAI) |
| **UI** | Streamlit |
| **API** | FastAPI + Uvicorn |
| **데이터 검증** | Pydantic v2 |
| **비동기 처리** | asyncio (병렬 답변 생성) |
| **평가 알고리즘** | LLM-as-a-Judge, Pairwise Comparison, Elo Rating |

---

## 기대 효과

- **동기 부여 극대화**: 경쟁 기반 학습으로 실전 긴장감 체득
- **실전 감각 배양**: 실제 취업 경쟁 상황을 반영한 현실적 면접 환경
- **맞춤형 피드백**: 항목별 세부 점수로 구체적 개선 방향 제시
- **확장 가능성**: 취업 플랫폼, 교육 서비스, 기업 내 인재 평가 도구로 확장 가능

---

## AI 윤리 준수 선언

본 프로젝트는 강의에서 학습한 **AI 윤리 가이드라인**을 준수합니다.

- 평가 과정의 투명성 확보 (근거 명시, 항목별 점수 공개)
- 데이터 편향 최소화 (성별·연령 등 인구통계 정보 평가 배제)
- 할루시네이션 위험 고지 (AI 평가는 참고용이며 최종 판단은 사용자)
- 개인정보 미수집 원칙 (답변 데이터 서버 저장 없음)
- AI 생성물 표시 (경쟁자 답변이 AI 생성임을 UI에 명시)

---

*본 프로젝트는 자연어처리(NLP) 강의의 중간 프로젝트로 제출되었습니다.*
*LLM Agent 설계, 평가 모델, 데이터셋 구축의 세 파트가 협력하여 완성한 End-to-End 시스템입니다.*
