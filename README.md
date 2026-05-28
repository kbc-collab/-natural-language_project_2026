# 🎯 AI 면접 경쟁 시스템 - LLM Agent 설계 파트

> **역할**: LLM Agent 설계 | **팀 파트**: 3파트 중 1번

## 📁 프로젝트 구조

```
word_project/
├── app/
│   ├── main.py                    # FastAPI 앱 진입점
│   ├── config.py                  # 환경 변수 관리
│   ├── agents/
│   │   ├── competitor_agent.py    # ✅ AI 경쟁자 에이전트 (3단계)
│   │   ├── interviewer_agent.py   # ✅ 면접관 에이전트
│   │   └── judge_agent.py         # 🔄 평가 에이전트 스텁 (평가팀 교체 예정)
│   ├── workflows/
│   │   └── interview_flow.py      # ✅ LangGraph 상태 머신
│   ├── schemas/
│   │   └── models.py              # ✅ 인터페이스 계약 (타 팀 공유)
│   └── api/
│       └── interview.py           # ✅ FastAPI 엔드포인트
├── prompts/
│   ├── competitor_level1.txt      # ✅ 초보 경쟁자 페르소나
│   ├── competitor_level2.txt      # ✅ 보통 경쟁자 페르소나
│   ├── competitor_level3.txt      # ✅ 우수 경쟁자 페르소나
│   └── interviewer.txt            # ✅ 면접관 페르소나
├── tests/
│   └── test_agents.py             # ✅ 단위 테스트
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 & 활성화
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 등 값을 채워주세요
```

### 2. 서버 실행
```bash
uvicorn app.main:app --reload
```
→ http://localhost:8000/docs 에서 Swagger UI 확인

### 3. 테스트 실행
```bash
pytest tests/ -v
```

## 🔌 타 팀 연동 가이드

### 평가팀 (Evaluation Model & Scoring)
`app/agents/judge_agent.py`의 `JudgeAgentStub.evaluate()` 메서드를 구현해주세요.

```python
# 입력: app/schemas/models.py → EvaluationRequest
# 출력: app/schemas/models.py → EvaluationResult

class JudgeAgent:
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        # LLM-as-a-Judge 로직 구현
        ...
```

완성 후 `judge_agent.py` 하단의 한 줄만 교체:
```python
from app.agents.judge_agent_impl import JudgeAgent as JudgeAgentStub
```

### 데이터셋팀 (Dataset & RAG Integration)
두 곳에 RAG 결과를 주입하면 됩니다:

1. **`app/agents/competitor_agent.py`** - `generate_all_answers()` 호출 시 `industry_context` 파라미터
2. **`app/api/interview.py`** - `submit_answer()` 내 `industry_context` 값

```python
# 현재 (하드코딩)
industry_context="최신 IT 산업 트렌드"

# 변경 후 (벡터 DB 검색 결과)
industry_context = await vector_db.search(query=target_job, top_k=3)
```

## 📡 API 엔드포인트 요약

| Method | URL | 설명 |
|--------|-----|------|
| `POST` | `/api/interview/sessions` | 면접 세션 시작 |
| `POST` | `/api/interview/sessions/{id}/answers` | 사용자 답변 제출 |
| `GET`  | `/api/interview/sessions/{id}/next-question` | 다음 질문 요청 |
| `GET`  | `/api/interview/sessions/{id}/report` | 최종 리포트 조회 |

## 🤝 팀원 인터페이스 계약

모든 데이터 형식은 `app/schemas/models.py`에 정의되어 있습니다.
**이 파일의 스키마는 팀 간 합의 없이 변경하지 마세요.**

- `EvaluationRequest` → 평가팀에게 전달하는 입력
- `EvaluationResult` → 평가팀이 반환해야 하는 출력  
- `QuestionContext` → 데이터셋팀이 RAG로 제공해야 하는 형식
