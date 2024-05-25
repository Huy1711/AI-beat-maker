from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    backend_url: str


settings = Settings()
