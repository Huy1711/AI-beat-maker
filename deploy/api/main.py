from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from schemas import DescriptionModeGenerateParam
from utils.suno.suno_client import SunoClient

# from utils.search.music_database_client import MusicDatabaseClient
# from utils.search.music_embedding_client import MusicEmbeddingClient


class Settings(BaseSettings):
    suno_cookie: str
    suno_session_id: str
    # music_embedding_url: str
    # music_database_url: str


settings = Settings()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

suno_client = SunoClient(cookie=settings.cookie, session_id=settings.session_id)
# music_embedding_client = MusicEmbeddingClient(settings.music_embedding_url)
# music_database_client = MusicDatabaseClient(settings.music_database_url)


@app.get("/")
async def root():
    return {"message": "Hello, this is AI beat maker project"}


@app.post("/generate")
async def generate_with_song_description(data: DescriptionModeGenerateParam):
    try:
        resp = await suno_client.generate(data.model_dump())
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@app.get("/feed/{aid}")
async def fetch_feed(aid: str):
    try:
        resp = await suno_client.get_feed(aid)
        return resp
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# @app.post("/search")
# async def search():
#     try:
#         pass
#     except Exception as e:
#         raise HTTPException(
#             detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )
