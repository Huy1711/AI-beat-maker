from typing import List

from pydantic import BaseModel, Field


class SunoInitParam(BaseModel):
    session_id: str = Field(..., description="Suno Session ID")
    cookie: str = Field(..., description="Suno Cookie")


class SunoGetSongsParam(BaseModel):
    ids: List[str] = Field(..., description="Suno Song IDs")


class DescriptionModeGenerateParam(BaseModel):
    """Generate with Song Description"""

    gpt_description_prompt: str
    make_instrumental: bool = False
    mv: str = Field(
        default="chirp-v3-0",
        description="model version, default: chirp-v3-0",
        examples=["chirp-v3-0"],
    )

    prompt: str = Field(
        default="",
        description="Placeholder, keep it as an empty string, do not modify it",
    )
