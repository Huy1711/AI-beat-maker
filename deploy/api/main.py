import logging
import traceback

from config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from schemas import DescriptionModeGenerateParam, SunoGetSongsParam, SunoInitParam
from utils.search.music_database_client import MusicDatabaseClient
from utils.search.music_embedding_client import MusicEmbeddingClient
from utils.search.summary import get_song_path, summary_result
from utils.suno.suno_client import SunoClient

logger = logging.getLogger("beat-maker-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.suno_client = None
app.music_embedding_client = MusicEmbeddingClient(settings.music_embedding_url)
app.music_database_client = MusicDatabaseClient(settings.music_database_url)


@app.get("/")
async def root():
    return JSONResponse({"message": "Hello, this is AI beat maker project"})


@app.post("/suno-connect")
async def generate(data: SunoInitParam):
    try:
        app.suno_client = SunoClient(**data.model_dump())
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return JSONResponse({"message": "Success"})


@app.post("/generate")
async def generate(data: DescriptionModeGenerateParam):
    logger.info("Generate request: Received")
    try:
        generate_results = await app.suno_client.generate_and_get_song(
            data.model_dump(), is_custom=False
        )
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    logger.info("Generate request: Done")
    return JSONResponse(content=generate_results)


@app.get("/suno-songs")
async def fetch_feed(data: SunoGetSongsParam):
    try:
        responses = await app.suno_client.get_song_by_ids(data.model_dump()["ids"])
    except Exception as e:
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return JSONResponse(content=responses)


@app.post("/search")
async def search(file: UploadFile = File(...)):
    logger.info(f"Search request: Received file {file.filename}")
    try:
        embeddings = app.music_embedding_client.get_embeddings(file)
        search_results = app.music_database_client.search_embeddings(embeddings)
        final_results = summary_result(search_results)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    finally:
        file.file.close()
    logger.info(f"Search request: Answered file {file.filename} : {final_results}")
    return JSONResponse(content=final_results)


@app.get("/songs/{file_id}")
async def get_song(file_id: str):
    try:
        audio_file = get_song_path(file_id)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise HTTPException(
            detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return FileResponse(audio_file)
