"""
면접관 에이전트 (Interviewer Agent)
======================================
면접 흐름을 주도하며 질문을 생성하는 에이전트입니다.
이전 대화를 기억하고 자연스러운 꼬리 질문을 합니다.
"""
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.schemas.models import QuestionOut, InterviewType
import uuid

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

# 직무별 기본 질문 시드 (데이터셋 팀이 Vector DB로 교체할 자리)
DEFAULT_QUESTION_SEEDS: dict[str, list[str]] = {
    "백엔드 개발자": [
        "자신이 경험한 가장 어려운 기술적 문제와 해결 과정을 말씀해 주세요.",
        "팀 프로젝트에서 갈등이 생겼을 때 어떻게 해결했나요?",
        "최근 공부하고 있는 기술이 있다면 소개해 주세요.",
    ],
    "마케터": [
        "성공적인 마케팅 캠페인 경험이 있다면 설명해 주세요.",
        "데이터를 기반으로 의사결정을 내린 경험을 말씀해 주세요.",
        "트렌드 변화에 빠르게 적응한 사례가 있나요?",
    ],
    "default": [
        "자기소개를 해주세요.",
        "지원 동기가 무엇인가요?",
        "본인의 강점과 약점은 무엇인가요?",
        "팀워크 경험을 말씀해 주세요.",
        "5년 후 목표는 무엇인가요?",
    ],
}


class InterviewerAgent:
    """
    면접관 에이전트.
    - 직무/면접 유형에 맞는 질문 생성
    - 이전 질문을 기억하여 중복 방지
    - 꼬리 질문(follow-up) 생성 가능
    """

    def __init__(
        self,
        target_job: str,
        interview_type: InterviewType = InterviewType.PERSONALITY,
        company_name: str = "당사",
    ):
        self.target_job = target_job
        self.interview_type = interview_type
        self.company_name = company_name
        self.previous_questions: list[str] = []

        self._prompt_template = (PROMPT_DIR / "interviewer.txt").read_text(encoding="utf-8")
        self._llm = ChatOpenAI(
            model=settings.interviewer_llm_model,
            temperature=0.6,
            max_tokens=300,
            api_key=settings.openai_api_key,
        )

    def _build_system_prompt(self) -> str:
        previous = "\n".join(
            [f"- {q}" for q in self.previous_questions]
        ) or "없음 (첫 번째 질문)"

        return self._prompt_template.format(
            company_name=self.company_name,
            interview_type=self.interview_type.value,
            target_job=self.target_job,
            previous_questions=previous,
        )

    async def generate_question(
        self,
        follow_up_context: str | None = None,
    ) -> QuestionOut:
        """
        다음 면접 질문을 생성합니다.

        Args:
            follow_up_context: 꼬리 질문 생성 시 이전 답변 내용 전달 (선택)

        Returns:
            QuestionOut: 질문 내용 및 메타데이터
        """
        system_prompt = self._build_system_prompt()
        user_message = (
            f"다음 면접 질문을 생성해주세요.\n이전 답변 요약: {follow_up_context}"
            if follow_up_context
            else "첫 번째 면접 질문을 생성해주세요."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        response = await self._llm.ainvoke(messages)
        question_text = response.content.strip()

        # 생성된 질문 기록 (중복 방지용)
        self.previous_questions.append(question_text)

        # 질문 의도 추출 (간단 버전; 추후 별도 LLM 호출로 정교화 가능)
        intent = self._infer_intent(question_text)

        return QuestionOut(
            question_id=str(uuid.uuid4()),
            content=question_text,
            intent=intent,
            time_limit_seconds=120,
        )

    def _infer_intent(self, question: str) -> str:
        """
        질문 텍스트를 보고 평가 의도를 간단히 추론합니다.
        (추후 LLM 호출 또는 사전 정의 사전으로 교체 가능)
        """
        keyword_intent_map = {
            "어려운": "문제 해결 능력 및 회복탄력성 평가",
            "갈등": "대인관계 및 커뮤니케이션 역량 평가",
            "성공": "성과 지향성 및 실행력 평가",
            "배운": "학습 능력 및 자기계발 의지 평가",
            "강점": "자기 인식 및 직무 적합성 평가",
            "약점": "자기 성찰 능력 및 성장 가능성 평가",
            "목표": "직업 가치관 및 장기 비전 평가",
            "팀": "협업 능력 및 리더십 평가",
            "자기소개": "전반적인 역량 및 커뮤니케이션 능력 평가",
            "지원": "직무 이해도 및 지원 동기의 진정성 평가",
        }
        for keyword, intent in keyword_intent_map.items():
            if keyword in question:
                return intent
        return "직무 적합성 및 역량 전반 평가"

    def get_fallback_question(self, index: int = 0) -> QuestionOut:
        """
        LLM 호출 실패 시 사용할 사전 정의 질문 (Fault Tolerance)
        """
        seeds = DEFAULT_QUESTION_SEEDS.get(
            self.target_job, DEFAULT_QUESTION_SEEDS["default"]
        )
        question_text = seeds[index % len(seeds)]
        self.previous_questions.append(question_text)
        return QuestionOut(
            question_id=str(uuid.uuid4()),
            content=question_text,
            intent=self._infer_intent(question_text),
            time_limit_seconds=120,
        )
