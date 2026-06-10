"""
앱 전역 설정 관리
pydantic-settings를 이용해 .env 파일에서 값을 자동 로드합니다.
"""
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class LLMProvider(str, Enum):
    """사용할 LLM 제공사"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    # 앱 기본 설정
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # LLM 제공사 선택 (openai / gemini / anthropic)
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, alias="LLM_PROVIDER")

    # API 키 (선택한 제공사의 키만 설정하면 됩니다)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # LLM 모델명 (제공사에 맞는 모델명으로 설정)
    competitor_llm_model: str = Field(default="gpt-4o-mini", alias="COMPETITOR_LLM_MODEL")
    judge_llm_model: str = Field(default="gpt-4o", alias="JUDGE_LLM_MODEL")
    interviewer_llm_model: str = Field(default="gpt-4o-mini", alias="INTERVIEWER_LLM_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1000, alias="LLM_MAX_TOKENS")

    # DB (평가팀 연동 시)
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/interview_db",
        alias="DATABASE_URL"
    )

    @model_validator(mode="after")
    def check_api_key(self) -> "Settings":
        """선택한 제공사의 API 키가 설정되어 있는지 확인합니다."""
        key_map = {
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GEMINI: self.gemini_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
        }
        if not key_map[self.llm_provider]:
            raise ValueError(
                f"LLM_PROVIDER='{self.llm_provider.value}'이지만 "
                f"해당 API 키가 .env에 설정되지 않았습니다."
            )
        return self

    model_config = {"env_file": ".env", "populate_by_name": True}


# 싱글턴 인스턴스 - 앱 전역에서 import하여 사용
settings = Settings()
