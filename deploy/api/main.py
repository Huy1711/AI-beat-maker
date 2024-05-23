import logging
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from schemas import DescriptionModeGenerateParam
from utils.search.music_database_client import MusicDatabaseClient
from utils.search.music_embedding_client import MusicEmbeddingClient
from utils.search.summary import format_result
from utils.suno.suno_client import SunoClient


class Settings(BaseSettings):
    suno_cookie: str
    suno_session_id: str
    music_embedding_url: str
    music_database_url: str


logger = logging.getLogger("beat-maker-api")
logging.basicConfig(level=logging.INFO)

settings = Settings()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

suno_client = SunoClient(
    cookie=settings.suno_cookie, session_id=settings.suno_session_id
)
music_embedding_client = MusicEmbeddingClient(settings.music_embedding_url)
music_database_client = MusicDatabaseClient(settings.music_database_url)


@app.get("/")
async def root():
    return {"message": "Hello, this is AI beat maker project"}


@app.post("/generate")
async def generate(data: DescriptionModeGenerateParam):
    try:
        generate_results = await suno_client.generate_and_get_song(
            data.model_dump(), is_custom=False
        )
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return JSONResponse(content=generate_results)


@app.post("/search")
async def search(file: UploadFile = File(...)):
    logger.info(f"Search request: Received file {file.filename}")
    try:
        embeddings = music_embedding_client.get_embeddings(file.file)
        print("embeddings", embeddings.shape)
        search_results = music_database_client.search_embeddings(embeddings)
        print("search_results", search_results)

        # final_results = format_result(search_results)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        file.file.close()
    return {"message": "No"}
    # logger.info(f"Search request: Answered file {file.filename} : {final_results}")
    # return final_results
