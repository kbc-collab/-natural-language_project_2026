"""
FastAPI 애플리케이션 진입점 (Entry Point)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.interview import router as interview_router

app = FastAPI(
    title="AI 면접 경쟁 시스템 - LLM Agent API",
    description="""
    ## 멀티 에이전트 기반 AI 면접 경쟁 시뮬레이터

    사용자가 AI 경쟁자들과 함께 면접에 참여하고,
    Judge Agent가 모든 답변을 평가하여 순위와 피드백을 제공합니다.

    ### 역할 분담
    - **LLM Agent 설계** (이 서비스): 경쟁자 에이전트, 면접관 에이전트, 워크플로우
    - **평가모델/Scoring** (평가팀): `judge_agent.py`의 `evaluate()` 메서드 구현
    - **데이터셋 실험 통합** (데이터팀): RAG를 통한 질문 데이터 및 산업 컨텍스트 주입
    """,
    version="0.1.0",
)

# CORS 설정 (프론트엔드 React/Next.js 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프로덕션에서는 실제 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(interview_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "running",
        "message": "AI 면접 경쟁 시스템 API",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
