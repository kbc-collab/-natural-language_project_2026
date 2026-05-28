"""
AI 경쟁자 에이전트 (Competitor Agent)
========================================
3가지 실력 레벨의 AI 면접 경쟁자를 생성합니다.

Level 1 (LOW)  - 초보: 짧고 추상적인 답변
Level 2 (MID)  - 보통: STAR 기법 시도, 수치 부족
Level 3 (HIGH) - 우수: 완벽한 STAR + 수치 + 전문용어
"""
import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.schemas.models import CompetitorLevel, CompetitorProfile, AnswerItem, RespondentType

# 프롬프트 파일 경로
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

# AI 경쟁자 캐릭터 정의
COMPETITOR_PROFILES: dict[CompetitorLevel, CompetitorProfile] = {
    CompetitorLevel.LOW: CompetitorProfile(
        id=1,
        name="김지훈 (첫 면접 준비생)",
        level=CompetitorLevel.LOW,
        description="이번이 첫 면접. 열정은 있지만 준비가 부족해요.",
    ),
    CompetitorLevel.MID: CompetitorProfile(
        id=2,
        name="박서연 (보통 준비생)",
        level=CompetitorLevel.MID,
        description="어느 정도 준비했지만 구체성이 아쉬워요.",
    ),
    CompetitorLevel.HIGH: CompetitorProfile(
        id=3,
        name="이준혁 (철저한 준비생)",
        level=CompetitorLevel.HIGH,
        description="데이터와 전문용어로 무장한 강력한 경쟁자예요.",
    ),
}


def _load_prompt(level: CompetitorLevel) -> str:
    """레벨에 맞는 프롬프트 템플릿 파일을 읽어옵니다."""
    filename = f"competitor_level{['low','mid','high'].index(level.value) + 1}.txt"
    prompt_path = PROMPT_DIR / filename
    return prompt_path.read_text(encoding="utf-8")


class CompetitorAgent:
    """
    단일 AI 경쟁자 에이전트.
    레벨에 따라 서로 다른 품질의 면접 답변을 생성합니다.
    """

    def __init__(self, level: CompetitorLevel):
        self.level = level
        self.profile = COMPETITOR_PROFILES[level]
        self._prompt_template = _load_prompt(level)

        # 레벨별 LLM 설정 차별화
        # LOW: temperature 높게(더 예측 불가), HIGH: temperature 낮게(더 정돈됨)
        temperature_map = {
            CompetitorLevel.LOW: 0.9,
            CompetitorLevel.MID: 0.7,
            CompetitorLevel.HIGH: 0.4,
        }
        self._llm = ChatOpenAI(
            model=settings.competitor_llm_model,
            temperature=temperature_map[level],
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key,
        )

    def _build_system_prompt(
        self,
        target_job: str,
        question_intent: str,
        industry_context: str = "최신 IT 산업 트렌드 및 디지털 전환",
    ) -> str:
        """프롬프트 템플릿에 변수를 채워 시스템 프롬프트를 생성합니다."""
        return self._prompt_template.format(
            target_job=target_job,
            question_intent=question_intent,
            industry_context=industry_context,  # Level 3에만 사용됨
        )

    async def generate_answer(
        self,
        question: str,
        target_job: str,
        question_intent: str,
        industry_context: str = "최신 IT 산업 트렌드",
    ) -> AnswerItem:
        """
        면접 질문에 대한 AI 경쟁자 답변을 비동기로 생성합니다.

        Args:
            question: 면접 질문 본문
            target_job: 지원 직무
            question_intent: 질문의 평가 의도
            industry_context: RAG로 주입할 산업 맥락 (데이터셋팀 연동 포인트)

        Returns:
            AnswerItem: 답변 내용 및 메타데이터
        """
        system_prompt = self._build_system_prompt(
            target_job=target_job,
            question_intent=question_intent,
            industry_context=industry_context,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]

        response = await self._llm.ainvoke(messages)
        return AnswerItem(
            respondent_type=RespondentType.AI_COMPETITOR,
            competitor_id=self.profile.id,
            competitor_level=self.level,
            content=response.content.strip(),
        )


class CompetitorAgentPool:
    """
    여러 AI 경쟁자 에이전트를 묶어 관리하는 풀(Pool).
    세션 시작 시 어떤 레벨 조합을 쓸지 선택하고,
    답변을 병렬로 생성합니다.
    """

    def __init__(self, levels: list[CompetitorLevel]):
        self.agents: list[CompetitorAgent] = [
            CompetitorAgent(level) for level in levels
        ]

    @property
    def profiles(self) -> list[CompetitorProfile]:
        """현재 풀에 포함된 경쟁자 프로필 목록"""
        return [agent.profile for agent in self.agents]

    async def generate_all_answers(
        self,
        question: str,
        target_job: str,
        question_intent: str,
        industry_context: str = "최신 IT 산업 트렌드",
    ) -> list[AnswerItem]:
        """
        모든 경쟁자의 답변을 병렬(asyncio.gather)로 생성합니다.
        LLM API 호출이 독립적이므로 동시에 요청해 응답 시간을 최소화합니다.
        """
        tasks = [
            agent.generate_answer(
                question=question,
                target_job=target_job,
                question_intent=question_intent,
                industry_context=industry_context,
            )
            for agent in self.agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        answers: list[AnswerItem] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Fallback: 에이전트 실패 시 기본 답변으로 대체
                print(f"[WARN] 경쟁자 {self.agents[i].profile.name} 답변 생성 실패: {result}")
                answers.append(_fallback_answer(self.agents[i]))
            else:
                answers.append(result)

        return answers


def _fallback_answer(agent: CompetitorAgent) -> AnswerItem:
    """LLM 호출 실패 시 사용할 기본 답변 (Fault Tolerance)"""
    fallback_texts = {
        CompetitorLevel.LOW: "열심히 노력하겠습니다. 최선을 다하겠습니다.",
        CompetitorLevel.MID: "이전 프로젝트에서 팀과 협력하여 문제를 해결한 경험이 있습니다. 앞으로도 그렇게 하겠습니다.",
        CompetitorLevel.HIGH: "해당 직무에서 요구하는 역량을 체계적으로 쌓아왔습니다. 구체적인 성과와 함께 기여하겠습니다.",
    }
    return AnswerItem(
        respondent_type=RespondentType.AI_COMPETITOR,
        competitor_id=agent.profile.id,
        competitor_level=agent.level,
        content=fallback_texts[agent.level],
    )
