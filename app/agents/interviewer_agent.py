"""
면접관 에이전트 (Interviewer Agent)
======================================
면접 흐름을 주도하며 질문을 생성하는 에이전트입니다.
이전 대화를 기억하고 자연스러운 꼬리 질문을 합니다.
데이터셋 기반 질문 시드와 평가 기준을 활용합니다.
"""
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.utils.llm_factory import create_llm
from app.utils.dataset_loader import dataset_loader
from app.schemas.models import QuestionOut, InterviewType
import uuid

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


class InterviewerAgent:
    """
    면접관 에이전트.
    - 직무/면접 유형에 맞는 질문 생성
    - 데이터셋 질문을 참고 시드 및 폴백으로 활용
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
        self._llm = create_llm(
            model=settings.interviewer_llm_model,
            temperature=0.6,
            max_tokens=300,
        )

        # 데이터셋에서 해당 면접 유형에 맞는 질문 목록 로드 (폴백 및 참고용)
        self._question_seeds: list[str] = dataset_loader.get_question_texts(
            interview_type.value
        )

    def _build_system_prompt(self) -> str:
        previous = (
            "\n".join([f"- {q}" for q in self.previous_questions])
            or "없음 (첫 번째 질문)"
        )

        # 아직 사용하지 않은 데이터셋 질문을 참고 예시로 제공 (최대 5개)
        used = set(self.previous_questions)
        available_seeds = [q for q in self._question_seeds if q not in used]
        sample_questions = (
            "\n".join([f"- {q}" for q in available_seeds[:5]])
            or "없음"
        )

        return self._prompt_template.format(
            company_name=self.company_name,
            interview_type=self.interview_type.value,
            target_job=self.target_job,
            previous_questions=previous,
            sample_questions=sample_questions,
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

        self.previous_questions.append(question_text)

        # 데이터셋에서 정확히 일치하는 질문의 평가 기준 조회
        criteria = dataset_loader.get_criteria(question_text)
        intent = self._infer_intent(question_text)

        return QuestionOut(
            question_id=str(uuid.uuid4()),
            content=question_text,
            intent=intent,
            criteria=criteria,
            time_limit_seconds=120,
        )

    def _infer_intent(self, question: str) -> str:
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
            "트렌드": "최신 기술 동향 파악 및 자기계발 의지 평가",
            "윤리": "AI 윤리 인식 및 원칙 기반 판단력 평가",
            "성능": "기술적 문제 해결 및 디버깅 능력 평가",
            "아키텍처": "시스템 설계 및 기술적 사고 능력 평가",
            "MLOps": "모델 운영 및 파이프라인 설계 능력 평가",
            "데이터": "데이터 처리 및 품질 관리 역량 평가",
        }
        for keyword, intent in keyword_intent_map.items():
            if keyword in question:
                return intent
        return "직무 적합성 및 역량 전반 평가"

    def get_fallback_question(self, index: int = 0) -> QuestionOut:
        """
        LLM 호출 실패 시 데이터셋 질문을 직접 사용합니다 (Fault Tolerance).
        사용한 질문은 제외하여 중복을 방지합니다.
        """
        used = set(self.previous_questions)
        available = [q for q in self._question_seeds if q not in used]

        if not available:
            # 데이터셋 질문 소진 시 기본 질문으로 대체
            available = [
                "자기소개를 해주세요.",
                "지원 동기가 무엇인가요?",
                "본인의 강점과 약점은 무엇인가요?",
            ]

        question_text = available[index % len(available)]
        criteria = dataset_loader.get_criteria(question_text)
        self.previous_questions.append(question_text)

        return QuestionOut(
            question_id=str(uuid.uuid4()),
            content=question_text,
            intent=self._infer_intent(question_text),
            criteria=criteria,
            time_limit_seconds=120,
        )
