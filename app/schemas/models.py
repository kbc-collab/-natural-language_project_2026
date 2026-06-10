"""
인터페이스 계약 (Interface Contract)
======================================
이 파일은 LLM Agent 파트 ↔ 평가팀 ↔ 데이터셋팀 간의 데이터 형식을 정의합니다.
다른 팀원이 이 스키마에 맞춰 입출력을 구현하면 됩니다.

[평가팀에게]
  - EvaluationRequest: Judge Agent에게 전달할 입력 형식
  - EvaluationResult: Judge Agent가 반환해야 할 출력 형식

[데이터셋팀에게]
  - QuestionContext: RAG로 주입할 질문 맥락 형식
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────
# 공통 열거형
# ──────────────────────────────────────────

class CompetitorLevel(str, Enum):
    """AI 경쟁자 실력 등급"""
    LOW = "low"       # 초보 (Level 1)
    MID = "mid"       # 보통 (Level 2)
    HIGH = "high"     # 우수 (Level 3)


class RespondentType(str, Enum):
    """답변 주체 구분"""
    USER = "USER"
    AI_COMPETITOR = "AI_COMPETITOR"


class InterviewType(str, Enum):
    """면접 유형"""
    PERSONALITY = "인성면접"
    TECHNICAL = "기술면접"
    HR = "HR면접"


class ComparisonResult(str, Enum):
    """Pairwise 비교 결과"""
    A_WINS = "A_WINS"
    B_WINS = "B_WINS"
    TIE = "TIE"


# ──────────────────────────────────────────
# 면접 세션 관련
# ──────────────────────────────────────────

class SessionStartRequest(BaseModel):
    """면접 세션 시작 요청 (프론트엔드 → API)"""
    user_id: str
    target_job: str = Field(..., example="백엔드 개발자")
    interview_type: InterviewType = InterviewType.PERSONALITY
    company_name: Optional[str] = Field(default=None, example="카카오")
    competitor_levels: list[CompetitorLevel] = Field(
        default=[CompetitorLevel.LOW, CompetitorLevel.HIGH],
        description="세션에 참여할 AI 경쟁자 레벨 목록"
    )


class SessionStartResponse(BaseModel):
    """면접 세션 시작 응답 (API → 프론트엔드)"""
    session_id: str
    first_question: QuestionOut
    competitors: list[CompetitorProfile]


# ──────────────────────────────────────────
# 질문 관련
# ──────────────────────────────────────────

class QuestionContext(BaseModel):
    """
    [데이터셋팀 인터페이스]
    RAG로 주입될 질문 맥락 데이터 형식입니다.
    데이터셋팀은 이 형식으로 벡터 DB 검색 결과를 반환해주세요.
    """
    question_text: str = Field(..., description="질문 본문")
    intent: str = Field(..., description="이 질문의 평가 의도")
    keywords: list[str] = Field(default=[], description="평가 핵심 키워드")
    example_good_answer: Optional[str] = Field(default=None, description="우수 답변 예시 (선택)")


class QuestionOut(BaseModel):
    """클라이언트에 전달할 질문 형식"""
    question_id: str
    content: str
    intent: str
    criteria: Optional[str] = Field(
        default=None,
        description="좋은 답변을 위한 평가 기준 (데이터셋 기반, 없으면 None)"
    )
    time_limit_seconds: int = 120


# ──────────────────────────────────────────
# 답변 관련
# ──────────────────────────────────────────

class UserAnswerSubmit(BaseModel):
    """사용자 답변 제출 (프론트엔드 → API)"""
    session_id: str
    question_id: str
    content: str
    response_time_seconds: int = Field(..., description="사용자가 답변 작성에 걸린 시간(초)")


class AnswerItem(BaseModel):
    """단일 답변 항목 (비교 시 사용)"""
    respondent_type: RespondentType
    competitor_id: Optional[int] = None          # AI일 때만 설정
    competitor_level: Optional[CompetitorLevel] = None
    content: str
    response_time_seconds: Optional[int] = None


# ──────────────────────────────────────────
# 평가 관련 (평가팀 인터페이스)
# ──────────────────────────────────────────

class EvaluationRequest(BaseModel):
    """
    [평가팀 인터페이스 - 입력]
    Judge Agent에게 전달되는 평가 요청 형식입니다.
    평가팀은 이 객체를 받아서 EvaluationResult를 반환하는
    judge_agent.evaluate() 메서드를 구현해주세요.
    """
    question_id: str
    question_content: str
    question_intent: str
    evaluation_criteria: Optional[str] = Field(
        default=None,
        description="좋은 답변을 위한 세부 평가 기준 (데이터셋 기반). "
                    "Judge Agent가 채점 루브릭으로 활용하세요."
    )
    answers: list[AnswerItem] = Field(
        ..., description="사용자 + AI 경쟁자 전체 답변 목록"
    )
    target_job: str


class ScoreBreakdown(BaseModel):
    """
    [평가팀 인터페이스 - 세부 점수]
    방사형 그래프(Radar Chart)에 사용될 항목별 점수 (0~100)
    """
    logic: int = Field(..., ge=0, le=100, description="논리성")
    professionalism: int = Field(..., ge=0, le=100, description="전문성")
    relevance: int = Field(..., ge=0, le=100, description="직무 적합성")
    clarity: int = Field(..., ge=0, le=100, description="명확성")
    concreteness: int = Field(..., ge=0, le=100, description="구체성 (수치/사례 포함 여부)")


class PairwiseComparison(BaseModel):
    """
    [평가팀 인터페이스 - Pairwise 비교]
    두 답변을 1:1로 비교한 결과입니다.
    N개의 답변이 있을 때 N*(N-1)/2 쌍이 생성됩니다.
    """
    answer_a_respondent: RespondentType
    answer_a_competitor_id: Optional[int] = None
    answer_b_respondent: RespondentType
    answer_b_competitor_id: Optional[int] = None
    result: ComparisonResult
    reasoning: str = Field(..., description="LLM이 판단한 비교 근거")


class EloConfig(BaseModel):
    """
    [평가팀 인터페이스 - Elo 계산 파라미터]
    Elo Rating 계산에 사용되는 설정값입니다.
    필요 시 평가팀이 조정할 수 있습니다.

    Elo 업데이트 공식:
      E_A = 1 / (1 + 10^((R_B - R_A) / 400))
      R_A_new = R_A + K * (S_A - E_A)
      S_A: 승=1.0, 무=0.5, 패=0.0
    """
    initial_score: float = Field(default=1000.0, description="참가자 초기 Elo 점수")
    k_factor: float = Field(default=32.0, description="점수 변동 민감도 (높을수록 변동 큼)")


class SingleEvaluation(BaseModel):
    """단일 답변에 대한 평가 결과"""
    respondent_type: RespondentType
    competitor_id: Optional[int] = None
    score_breakdown: ScoreBreakdown
    total_score: float
    elo_score: float = Field(default=1000.0, description="Pairwise 비교 후 산정된 Elo 점수")
    feedback: str = Field(..., description="해당 답변의 구체적 피드백")
    rank: int = Field(..., description="이 질문에서의 Elo 기준 순위 (1위가 최고)")


class EvaluationResult(BaseModel):
    """
    [평가팀 인터페이스 - 출력]
    Judge Agent가 반환해야 할 전체 평가 결과 형식입니다.

    평가 흐름:
      1. pairwise_comparisons: 모든 답변 쌍을 1:1 비교
      2. elo_config 기준으로 각 참가자 Elo 점수 업데이트
      3. evaluations: 최종 Elo 점수 및 rank 반영
    """
    question_id: str
    evaluations: list[SingleEvaluation]
    pairwise_comparisons: list[PairwiseComparison] = Field(
        default=[],
        description="1:1 비교 결과 목록 (N*(N-1)/2 쌍)"
    )
    elo_config: EloConfig = Field(
        default_factory=EloConfig,
        description="Elo 계산에 사용된 파라미터"
    )
    comparative_feedback: str = Field(
        ...,
        description="비교 분석 피드백 예: '경쟁자 A는 수치를 언급했지만 사용자는...' "
    )
    improvement_tip: str = Field(..., description="사용자를 위한 구체적 개선 제안")


# ──────────────────────────────────────────
# AI 경쟁자 프로필
# ──────────────────────────────────────────

class CompetitorProfile(BaseModel):
    """AI 경쟁자 캐릭터 정보"""
    id: int
    name: str
    level: CompetitorLevel
    description: str


# ──────────────────────────────────────────
# 최종 리포트
# ──────────────────────────────────────────

class SessionReport(BaseModel):
    """세션 종료 후 최종 리포트"""
    session_id: str
    user_id: str
    overall_rank: int
    average_score: float
    question_results: list[EvaluationResult]
    final_feedback: str
    growth_areas: list[str] = Field(description="보완이 필요한 영역 목록")
