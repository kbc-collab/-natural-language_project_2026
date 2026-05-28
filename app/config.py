"""
앱 전역 설정 관리
pydantic-settings를 이용해 .env 파일에서 값을 자동 로드합니다.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # 앱 기본 설정
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # LLM 모델 설정
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

    model_config = {"env_file": ".env", "populate_by_name": True}


# 싱글턴 인스턴스 - 앱 전역에서 import하여 사용
settings = Settings()
