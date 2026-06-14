"""
평가 에이전트 스텁 (Judge Agent Stub)
========================================
⚠️  이 파일은 평가팀(Evaluation Model & Scoring)이 구현해야 합니다.
    LLM Agent 파트에서는 인터페이스(입출력 형식)만 정의합니다.

[평가팀에게 전달하는 계약]
  입력: EvaluationRequest  (app/schemas/models.py 참조)
  출력: EvaluationResult   (app/schemas/models.py 참조)

  아래 JudgeAgentStub.evaluate() 메서드를 실제 LLM-as-a-Judge 로직으로
  교체하시면 됩니다. 메서드 시그니처(입출력 타입)는 유지해주세요.

[구현 힌트]
  - GPT-4o 등 강력한 모델을 judge_llm_model 설정으로 사용하세요.
  - 각 답변을 동시에 평가하되, 서로를 비교하는 '비교 평가 프롬프트'를 설계하세요.
  - score_breakdown의 각 항목(논리성, 전문성, 관련성, 명확성, 구체성)은
    0~100 정수값이어야 합니다 (Radar Chart 렌더링용).
"""
import asyncio
import random
from app.schemas.models import (
    EvaluationRequest,
    EvaluationResult,
    SingleEvaluation,
    ScoreBreakdown,
    PairwiseComparison,
    ComparisonResult,
    EloConfig,
    RespondentType,
)
from app.config import settings


class JudgeAgentStub:
    """
    평가 에이전트 스텁 (Mock 구현체).
    실제 LLM 평가 로직이 완성되기 전까지 임의 점수로 동작합니다.
    평가팀은 이 클래스를 JudgeAgent로 교체해 주세요.
    """

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        [평가팀 구현 대상]
        복수의 답변을 읽고 각각에 대한 세부 점수와 피드백을 반환합니다.

        Args:
            request: EvaluationRequest - 질문 + 모든 답변자 답변 목록

        Returns:
            EvaluationResult - 전체 순위, 세부 점수, 비교 피드백
        """
        # ── 스텁: 더미 점수 생성 (평가팀이 실제 LLM 로직으로 교체) ──
        await asyncio.sleep(0.1)  # 실제 LLM 응답 시간 시뮬레이션

        evaluations: list[SingleEvaluation] = []

        # 답변 유형별 점수 범위 설정 (스텁 전용)
        score_ranges = {
            RespondentType.USER: (50, 75),
            RespondentType.AI_COMPETITOR: None,  # 레벨별로 별도 처리
        }

        from app.schemas.models import CompetitorLevel
        level_ranges = {
            CompetitorLevel.LOW: (20, 45),
            CompetitorLevel.MID: (45, 70),
            CompetitorLevel.HIGH: (75, 95),
        }

        for answer in request.answers:
            if answer.respondent_type == RespondentType.USER:
                lo, hi = score_ranges[RespondentType.USER]
            else:
                lo, hi = level_ranges.get(answer.competitor_level, (50, 70))

            scores = ScoreBreakdown(
                logic=random.randint(lo, hi),
                professionalism=random.randint(lo, hi),
                relevance=random.randint(lo, hi),
                clarity=random.randint(lo, hi),
                concreteness=random.randint(lo, hi),
            )
            total = round(
                (scores.logic + scores.professionalism + scores.relevance
                 + scores.clarity + scores.concreteness) / 5, 1
            )
            evaluations.append(SingleEvaluation(
                respondent_type=answer.respondent_type,
                competitor_id=answer.competitor_id,
                score_breakdown=scores,
                total_score=total,
                feedback="[스텁] 평가팀이 실제 LLM 피드백으로 교체 예정입니다.",
                rank=0,  # 아래에서 정렬 후 재할당
            ))

        # 스텁 Pairwise 비교 결과 생성 (무작위)
        pairwise: list[PairwiseComparison] = []
        for i in range(len(evaluations)):
            for j in range(i + 1, len(evaluations)):
                a, b = evaluations[i], evaluations[j]
                if a.total_score > b.total_score:
                    result = ComparisonResult.A_WINS
                elif a.total_score < b.total_score:
                    result = ComparisonResult.B_WINS
                else:
                    result = ComparisonResult.TIE
                pairwise.append(PairwiseComparison(
                    answer_a_respondent=a.respondent_type,
                    answer_a_competitor_id=a.competitor_id,
                    answer_b_respondent=b.respondent_type,
                    answer_b_competitor_id=b.competitor_id,
                    result=result,
                    reasoning="[스텁] 평가팀이 LLM 비교 근거로 교체 예정입니다.",
                ))

        # Pairwise 결과 기반 Elo 점수 계산
        elo_config = EloConfig()
        elo_scores = {ev.respondent_type.value + str(ev.competitor_id): elo_config.initial_score
                      for ev in evaluations}

        def _elo_key(ev: SingleEvaluation) -> str:
            return ev.respondent_type.value + str(ev.competitor_id)

        for pair in pairwise:
            key_a = pair.answer_a_respondent.value + str(pair.answer_a_competitor_id)
            key_b = pair.answer_b_respondent.value + str(pair.answer_b_competitor_id)
            r_a, r_b = elo_scores[key_a], elo_scores[key_b]
            e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
            e_b = 1 - e_a
            s_a, s_b = (1.0, 0.0) if pair.result == ComparisonResult.A_WINS else \
                       (0.0, 1.0) if pair.result == ComparisonResult.B_WINS else \
                       (0.5, 0.5)
            elo_scores[key_a] += elo_config.k_factor * (s_a - e_a)
            elo_scores[key_b] += elo_config.k_factor * (s_b - e_b)

        for ev in evaluations:
            ev.elo_score = round(elo_scores[_elo_key(ev)], 1)

        # Elo 점수 기준 순위 산정
        evaluations.sort(key=lambda e: e.elo_score, reverse=True)
        for i, ev in enumerate(evaluations):
            ev.rank = i + 1

        return EvaluationResult(
            question_id=request.question_id,
            evaluations=evaluations,
            pairwise_comparisons=pairwise,
            elo_config=elo_config,
            comparative_feedback=(
                "[스텁] 비교 분석 피드백 — 평가팀이 LLM-as-a-Judge 로직으로 교체 예정입니다."
            ),
            improvement_tip=(
                "[스텁] 개선 제안 — 평가팀이 구체적 제안 생성 로직으로 교체 예정입니다."
            ),
        )


# 실제 배포 시에는 아래 한 줄만 교체하면 됩니다:
#   from app.agents.judge_agent_impl import JudgeAgent as JudgeAgentStub
judge_agent = JudgeAgentStub()
