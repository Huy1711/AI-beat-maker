from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    music_embedding_url: str
    music_database_url: str


settings = Settings()
