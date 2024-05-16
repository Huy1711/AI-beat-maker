import json

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# @app.post("/generate")
# async def generate():
#     try:
#         pass
#     except Exception as e:
#         raise HTTPException(
#             detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )


# @app.post("/search")
# async def search():
#     try:
#         pass
#     except Exception as e:
#         raise HTTPException(
#             detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )