"""
면접 워크플로우 (LangGraph State Machine)
==========================================
LangGraph를 이용해 면접 전체 흐름을 상태 기반으로 제어합니다.

[상태 전이 흐름]
  START
    ↓
  generate_question     ← 면접관 에이전트가 질문 생성
    ↓
  collect_user_answer   ← 사용자 답변 대기 (API 호출로 외부에서 주입)
    ↓
  generate_competitor_answers  ← AI 경쟁자 병렬 답변 생성
    ↓
  evaluate_answers      ← Judge Agent 평가 (스텁 → 평가팀 교체)
    ↓
  check_continue        ← 질문 수 초과 여부 판단
    ↓ (더 있으면 generate_question으로 루프)
  generate_report       ← 최종 리포트 생성
    ↓
  END
"""
import uuid
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.agents.interviewer_agent import InterviewerAgent
from app.agents.competitor_agent import CompetitorAgentPool
from app.agents.judge_agent import judge_agent
from app.utils.dataset_loader import dataset_loader
from app.schemas.models import (
    QuestionOut,
    AnswerItem,
    EvaluationRequest,
    EvaluationResult,
    RespondentType,
    CompetitorLevel,
    InterviewType,
    SessionReport,
)

MAX_QUESTIONS = 5  # 세션당 최대 질문 수


# ──────────────────────────────────────────
# 워크플로우 상태 정의
# ──────────────────────────────────────────

class InterviewState(TypedDict):
    """LangGraph가 추적할 면접 세션 전체 상태"""
    session_id: str
    user_id: str
    target_job: str
    interview_type: str
    company_name: str
    competitor_levels: list[str]

    # 진행 상태
    current_question_index: int
    questions: list[QuestionOut]
    all_answers: dict[str, list[AnswerItem]]    # question_id → 답변 목록
    all_evaluations: list[EvaluationResult]

    # 사용자 입력 (외부에서 주입)
    pending_user_answer: AnswerItem | None

    # 상태 플래그
    is_complete: bool
    error: str | None


# ──────────────────────────────────────────
# 노드 함수 (각 단계 처리 로직)
# ──────────────────────────────────────────

async def node_generate_question(state: InterviewState) -> dict:
    """
    [노드 1] 면접관 에이전트가 다음 질문을 생성합니다.
    이미 생성된 질문 목록을 참고하여 중복을 방지합니다.
    """
    # 면접관 에이전트 (상태 기반 재생성 — 프로덕션에선 캐시 권장)
    interviewer = InterviewerAgent(
        target_job=state["target_job"],
        interview_type=InterviewType(state["interview_type"]),
        company_name=state["company_name"],
    )
    # 이전 질문들을 면접관에게 주입
    interviewer.previous_questions = [q.content for q in state["questions"]]

    try:
        question = await interviewer.generate_question()
    except Exception as e:
        print(f"[WARN] 질문 생성 실패, fallback 사용: {e}")
        question = interviewer.get_fallback_question(state["current_question_index"])

    updated_questions = state["questions"] + [question]
    return {"questions": updated_questions}


async def node_generate_competitor_answers(state: InterviewState) -> dict:
    """
    [노드 2] 현재 질문에 대해 모든 AI 경쟁자 답변을 병렬 생성합니다.
    """
    current_question = state["questions"][state["current_question_index"]]

    competitor_pool = CompetitorAgentPool(
        levels=[CompetitorLevel(lvl) for lvl in state["competitor_levels"]]
    )

    competitor_answers = await competitor_pool.generate_all_answers(
        question=current_question.content,
        target_job=state["target_job"],
        question_intent=current_question.intent,
        # 데이터셋팀 RAG 연동 포인트: industry_context를 벡터 DB 검색 결과로 교체
        industry_context="최신 IT 산업 트렌드 및 AI 전환 가속화",
        interview_type_value=state["interview_type"],
    )

    # 사용자 답변 + 경쟁자 답변 합치기
    user_answer = state["pending_user_answer"]
    all_answers_for_question = ([user_answer] if user_answer else []) + competitor_answers

    updated_all_answers = dict(state["all_answers"])
    updated_all_answers[current_question.question_id] = all_answers_for_question

    return {
        "all_answers": updated_all_answers,
        "pending_user_answer": None,  # 처리 완료 후 초기화
    }


async def node_evaluate_answers(state: InterviewState) -> dict:
    """
    [노드 3] Judge Agent가 현재 질문의 모든 답변을 평가합니다.
    평가팀이 judge_agent.evaluate()를 실제 구현으로 교체하면 자동으로 적용됩니다.
    """
    current_question = state["questions"][state["current_question_index"]]
    answers = state["all_answers"].get(current_question.question_id, [])

    # 데이터셋에서 평가 기준 조회 (질문이 데이터셋에 없으면 None)
    criteria = current_question.criteria or dataset_loader.get_criteria(current_question.content)

    eval_request = EvaluationRequest(
        question_id=current_question.question_id,
        question_content=current_question.content,
        question_intent=current_question.intent,
        evaluation_criteria=criteria,
        answers=answers,
        target_job=state["target_job"],
    )

    try:
        evaluation = await judge_agent.evaluate(eval_request)
    except Exception as e:
        print(f"[ERROR] 평가 실패: {e}")
        return {"error": f"평가 실패: {e}"}

    updated_evaluations = state["all_evaluations"] + [evaluation]
    return {
        "all_evaluations": updated_evaluations,
        "current_question_index": state["current_question_index"] + 1,
    }


def node_check_continue(state: InterviewState) -> str:
    """
    [조건 분기] 다음 질문으로 계속할지, 종료할지 결정합니다.
    LangGraph의 conditional_edge에서 사용합니다.
    """
    if state["current_question_index"] >= MAX_QUESTIONS:
        return "generate_report"
    if state.get("error"):
        return "generate_report"
    return "generate_question"


async def node_generate_report(state: InterviewState) -> dict:
    """
    [노드 4] 세션 전체 결과를 집계하여 최종 리포트를 생성합니다.
    """
    all_evals = state["all_evaluations"]

    if not all_evals:
        return {"is_complete": True}

    # 사용자 평균 점수 계산
    user_scores = []
    for ev in all_evals:
        for single in ev.evaluations:
            if single.respondent_type == RespondentType.USER:
                user_scores.append(single.total_score)

    average_score = round(sum(user_scores) / len(user_scores), 1) if user_scores else 0.0

    # 사용자 순위 (마지막 질문 기준 간단 계산)
    last_eval = all_evals[-1]
    user_rank = next(
        (s.rank for s in last_eval.evaluations if s.respondent_type == RespondentType.USER),
        len(last_eval.evaluations),
    )

    print(f"[리포트] 세션 {state['session_id']} 완료 — 평균 점수: {average_score}, 순위: {user_rank}")
    return {"is_complete": True}


# ──────────────────────────────────────────
# LangGraph 그래프 빌드
# ──────────────────────────────────────────

def build_interview_graph() -> StateGraph:
    """
    면접 워크플로우 LangGraph를 빌드하여 반환합니다.
    main.py에서 한 번만 빌드하고 재사용합니다.
    """
    builder = StateGraph(InterviewState)

    # 노드 등록
    builder.add_node("generate_question", node_generate_question)
    builder.add_node("generate_competitor_answers", node_generate_competitor_answers)
    builder.add_node("evaluate_answers", node_evaluate_answers)
    builder.add_node("generate_report", node_generate_report)

    # 엣지 연결 (선형 흐름)
    builder.set_entry_point("generate_question")
    builder.add_edge("generate_question", "generate_competitor_answers")
    builder.add_edge("generate_competitor_answers", "evaluate_answers")

    # 조건 분기: 질문 수 초과 시 리포트, 아니면 다음 질문
    builder.add_conditional_edges(
        "evaluate_answers",
        node_check_continue,
        {
            "generate_question": "generate_question",
            "generate_report": "generate_report",
        },
    )
    builder.add_edge("generate_report", END)

    return builder.compile()


# 앱 시작 시 한 번 컴파일
interview_graph = build_interview_graph()


def create_initial_state(
    user_id: str,
    target_job: str,
    interview_type: str,
    company_name: str,
    competitor_levels: list[str],
) -> InterviewState:
    """새 면접 세션의 초기 상태를 생성합니다."""
    return InterviewState(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        target_job=target_job,
        interview_type=interview_type,
        company_name=company_name,
        competitor_levels=competitor_levels,
        current_question_index=0,
        questions=[],
        all_answers={},
        all_evaluations=[],
        pending_user_answer=None,
        is_complete=False,
        error=None,
    )
