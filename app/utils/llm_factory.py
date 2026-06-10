"""
LLM 팩토리 (LLM Factory)
==========================
LLM_PROVIDER 설정에 따라 OpenAI / Gemini / Anthropic 중 하나를 반환합니다.
에이전트는 이 함수만 호출하면 되고, 제공사 변경 시 .env만 수정하면 됩니다.
"""
from langchain_core.language_models import BaseChatModel
from app.config import settings, LLMProvider


def create_llm(model: str, temperature: float, max_tokens: int) -> BaseChatModel:
    """
    설정된 LLM_PROVIDER에 맞는 LangChain Chat 모델을 반환합니다.

    Args:
        model: 모델명 (.env의 *_LLM_MODEL 값)
        temperature: 생성 다양성 (0.0 ~ 1.0)
        max_tokens: 최대 출력 토큰 수

    Returns:
        LangChain BaseChatModel (ainvoke 인터페이스 동일)
    """
    provider = settings.llm_provider

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.openai_api_key,
        )

    if provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=settings.gemini_api_key,
        )

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.anthropic_api_key,
        )

    raise ValueError(f"지원하지 않는 LLM_PROVIDER: {provider}")
