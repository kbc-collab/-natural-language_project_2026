"""
에이전트 단위 테스트
======================
실제 OpenAI API 없이도 구조와 인터페이스를 검증하는 테스트입니다.
(LLM 호출 부분은 Mock 처리)
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.schemas.models import (
    CompetitorLevel,
    RespondentType,
    AnswerItem,
    EvaluationRequest,
    SessionStartRequest,
    InterviewType,
)


# ──────────────────────────────────────────
# 경쟁자 에이전트 테스트
# ──────────────────────────────────────────

class TestCompetitorAgent:

    @pytest.mark.asyncio
    async def test_competitor_agent_returns_answer_item(self):
        """CompetitorAgent가 AnswerItem 형식으로 답변을 반환하는지 확인"""
        from app.agents.competitor_agent import CompetitorAgent

        mock_response = MagicMock()
        mock_response.content = "열심히 하겠습니다. 최선을 다하겠습니다."

        with patch("app.agents.competitor_agent.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_response)

            agent = CompetitorAgent(CompetitorLevel.LOW)
            result = await agent.generate_answer(
                question="자기소개를 해주세요.",
                target_job="백엔드 개발자",
                question_intent="커뮤니케이션 능력 평가",
            )

        assert isinstance(result, AnswerItem)
        assert result.respondent_type == RespondentType.AI_COMPETITOR
        assert result.competitor_level == CompetitorLevel.LOW
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_competitor_pool_parallel_generation(self):
        """CompetitorAgentPool이 여러 에이전트 답변을 병렬 생성하는지 확인"""
        from app.agents.competitor_agent import CompetitorAgentPool

        mock_response = MagicMock()
        mock_response.content = "테스트 답변입니다."

        with patch("app.agents.competitor_agent.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_response)

            pool = CompetitorAgentPool([CompetitorLevel.LOW, CompetitorLevel.HIGH])
            results = await pool.generate_all_answers(
                question="강점을 말씀해주세요.",
                target_job="백엔드 개발자",
                question_intent="자기 인식 평가",
            )

        assert len(results) == 2
        assert all(isinstance(r, AnswerItem) for r in results)

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """LLM 호출 실패 시 Fallback 답변이 반환되는지 확인"""
        from app.agents.competitor_agent import CompetitorAgentPool

        with patch("app.agents.competitor_agent.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(side_effect=Exception("API 오류"))

            pool = CompetitorAgentPool([CompetitorLevel.LOW])
            results = await pool.generate_all_answers(
                question="팀워크 경험을 말해주세요.",
                target_job="마케터",
                question_intent="협업 능력 평가",
            )

        # Fallback이 동작하면 빈 리스트가 아니어야 함
        assert len(results) == 1
        assert results[0].respondent_type == RespondentType.AI_COMPETITOR


# ──────────────────────────────────────────
# 스키마 유효성 테스트
# ──────────────────────────────────────────

class TestSchemas:

    def test_session_start_request_validation(self):
        """SessionStartRequest 스키마 유효성 검증"""
        req = SessionStartRequest(
            user_id="user-001",
            target_job="백엔드 개발자",
            interview_type=InterviewType.TECHNICAL,
            company_name="카카오",
            competitor_levels=[CompetitorLevel.LOW, CompetitorLevel.HIGH],
        )
        assert req.target_job == "백엔드 개발자"
        assert len(req.competitor_levels) == 2

    def test_evaluation_request_with_multiple_answers(self):
        """EvaluationRequest가 복수 답변을 올바르게 담는지 확인"""
        answers = [
            AnswerItem(respondent_type=RespondentType.USER, content="사용자 답변"),
            AnswerItem(
                respondent_type=RespondentType.AI_COMPETITOR,
                competitor_id=1,
                competitor_level=CompetitorLevel.LOW,
                content="AI 답변 1",
            ),
            AnswerItem(
                respondent_type=RespondentType.AI_COMPETITOR,
                competitor_id=3,
                competitor_level=CompetitorLevel.HIGH,
                content="AI 답변 2",
            ),
        ]
        req = EvaluationRequest(
            question_id="q-001",
            question_content="자기소개를 해주세요.",
            question_intent="커뮤니케이션 능력 평가",
            answers=answers,
            target_job="백엔드 개발자",
        )
        assert len(req.answers) == 3


# ──────────────────────────────────────────
# Judge Agent 스텁 테스트
# ──────────────────────────────────────────

class TestJudgeAgentStub:

    @pytest.mark.asyncio
    async def test_stub_returns_evaluation_result(self):
        """JudgeAgentStub이 EvaluationResult를 올바르게 반환하는지 확인"""
        from app.agents.judge_agent import JudgeAgentStub
        from app.schemas.models import EvaluationResult

        stub = JudgeAgentStub()
        request = EvaluationRequest(
            question_id="q-test",
            question_content="강점이 뭔가요?",
            question_intent="자기 인식 평가",
            answers=[
                AnswerItem(respondent_type=RespondentType.USER, content="논리적 사고입니다."),
                AnswerItem(
                    respondent_type=RespondentType.AI_COMPETITOR,
                    competitor_id=1,
                    competitor_level=CompetitorLevel.LOW,
                    content="열심히 하는 것입니다.",
                ),
            ],
            target_job="백엔드 개발자",
        )

        result = await stub.evaluate(request)

        assert isinstance(result, EvaluationResult)
        assert len(result.evaluations) == 2
        assert all(0 < e.rank <= 2 for e in result.evaluations)
        # 순위가 중복되지 않는지 확인
        ranks = [e.rank for e in result.evaluations]
        assert len(set(ranks)) == len(ranks)
        # Pairwise 비교: 2명이면 1쌍
        assert len(result.pairwise_comparisons) == 1
        # Elo 점수가 초기값(1000)에서 변동됐는지 확인
        elo_scores = [e.elo_score for e in result.evaluations]
        assert any(s != 1000.0 for s in elo_scores)
