"""
면접 API 라우터
==================
FastAPI 엔드포인트 정의.
프론트엔드 ↔ 백엔드 통신 인터페이스입니다.
"""
import asyncio
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.schemas.models import (
    SessionStartRequest,
    SessionStartResponse,
    UserAnswerSubmit,
    EvaluationResult,
    AnswerItem,
    RespondentType,
    CompetitorLevel,
)
from app.agents.competitor_agent import CompetitorAgentPool, COMPETITOR_PROFILES
from app.agents.interviewer_agent import InterviewerAgent
from app.agents.judge_agent import judge_agent
from app.schemas.models import EvaluationRequest

router = APIRouter(prefix="/api/interview", tags=["Interview"])

# 인메모리 세션 스토어 (프로덕션에서는 Redis 또는 PostgreSQL로 교체)
# key: session_id, value: 세션 상태 dict
_sessions: dict[str, dict] = {}


# ──────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────

@router.post("/sessions", response_model=SessionStartResponse, summary="면접 세션 시작")
async def start_session(request: SessionStartRequest):
    """
    새 면접 세션을 시작합니다.
    - 면접관 에이전트가 첫 번째 질문을 생성합니다.
    - 세션에 참여할 AI 경쟁자 프로필을 반환합니다.
    """
    session_id = str(uuid.uuid4())

    # 면접관 에이전트 초기화 & 첫 질문 생성
    interviewer = InterviewerAgent(
        target_job=request.target_job,
        interview_type=request.interview_type,
        company_name=request.company_name or "당사",
    )

    try:
        first_question = await interviewer.generate_question()
    except Exception as e:
        print(f"[WARN] 첫 질문 생성 실패, fallback 사용: {e}")
        first_question = interviewer.get_fallback_question(0)

    # 세션 상태 저장
    _sessions[session_id] = {
        "session_id": session_id,
        "user_id": request.user_id,
        "target_job": request.target_job,
        "interview_type": request.interview_type,
        "company_name": request.company_name or "당사",
        "competitor_levels": request.competitor_levels,
        "questions": [first_question],
        "question_index": 0,
        "all_answers": {},
        "all_evaluations": [],
        "interviewer": interviewer,
    }

    competitors = [
        COMPETITOR_PROFILES[level] for level in request.competitor_levels
    ]

    return SessionStartResponse(
        session_id=session_id,
        first_question=first_question,
        competitors=competitors,
    )


@router.post(
    "/sessions/{session_id}/answers",
    response_model=EvaluationResult,
    summary="사용자 답변 제출 및 AI 경쟁자 답변 생성"
)
async def submit_answer(session_id: str, body: UserAnswerSubmit):
    """
    사용자의 답변을 제출받고:
    1. AI 경쟁자 답변을 병렬 생성합니다.
    2. Judge Agent가 모든 답변을 평가합니다.
    3. 평가 결과를 반환합니다.

    프론트엔드는 이 API를 호출한 뒤 "AI 경쟁자가 답변 중..." UX를 보여주면 됩니다.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 현재 질문 가져오기
    questions = session["questions"]
    question_index = session["question_index"]
    if question_index >= len(questions):
        raise HTTPException(status_code=400, detail="모든 질문이 완료되었습니다.")

    current_question = questions[question_index]

    # 1) 사용자 답변 객체 생성
    user_answer = AnswerItem(
        respondent_type=RespondentType.USER,
        content=body.content,
        response_time_seconds=body.response_time_seconds,
    )

    # 2) AI 경쟁자 답변 병렬 생성
    competitor_pool = CompetitorAgentPool(
        levels=session["competitor_levels"]
    )
    competitor_answers = await competitor_pool.generate_all_answers(
        question=current_question.content,
        target_job=session["target_job"],
        question_intent=current_question.intent,
        # 데이터셋팀 RAG 연동 포인트 ↓
        industry_context="최신 IT 산업 트렌드",
    )

    all_answers = [user_answer] + competitor_answers
    session["all_answers"][current_question.question_id] = all_answers

    # 3) Judge Agent 평가
    eval_request = EvaluationRequest(
        question_id=current_question.question_id,
        question_content=current_question.content,
        question_intent=current_question.intent,
        answers=all_answers,
        target_job=session["target_job"],
    )
    evaluation = await judge_agent.evaluate(eval_request)
    session["all_evaluations"].append(evaluation)
    session["question_index"] += 1

    return evaluation


@router.get(
    "/sessions/{session_id}/next-question",
    response_model=dict,
    summary="다음 질문 가져오기"
)
async def get_next_question(session_id: str):
    """
    현재 평가 완료 후 다음 질문을 요청합니다.
    더 이상 질문이 없으면 is_complete: true를 반환합니다.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    MAX_QUESTIONS = 5
    q_index = session["question_index"]

    if q_index >= MAX_QUESTIONS:
        return {"is_complete": True, "question": None}

    interviewer: InterviewerAgent = session["interviewer"]
    try:
        next_question = await interviewer.generate_question()
    except Exception as e:
        print(f"[WARN] 다음 질문 생성 실패, fallback: {e}")
        next_question = interviewer.get_fallback_question(q_index)

    session["questions"].append(next_question)

    return {"is_complete": False, "question": next_question.model_dump()}


@router.get(
    "/sessions/{session_id}/report",
    summary="최종 리포트 조회"
)
async def get_report(session_id: str):
    """
    세션의 최종 리포트를 반환합니다.
    전체 질문별 평가 결과와 종합 분석이 포함됩니다.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    evaluations = session["all_evaluations"]
    if not evaluations:
        raise HTTPException(status_code=400, detail="아직 완료된 평가가 없습니다.")

    # 사용자 평균 점수
    user_scores = []
    for ev in evaluations:
        for single in ev.evaluations:
            if single.respondent_type == RespondentType.USER:
                user_scores.append(single.total_score)

    avg = round(sum(user_scores) / len(user_scores), 1) if user_scores else 0.0

    # 마지막 질문 기준 순위
    last_rank = next(
        (s.rank for s in evaluations[-1].evaluations
         if s.respondent_type == RespondentType.USER),
        "-"
    )

    return {
        "session_id": session_id,
        "average_score": avg,
        "overall_rank": last_rank,
        "total_questions": len(evaluations),
        "evaluations": [ev.model_dump() for ev in evaluations],
        "message": "세션이 완료되었습니다. 상세 리포트를 확인하세요.",
    }
