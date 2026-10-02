"""Local Laya inference using the TypeSafe System One HTTP contract.

Reference: https://docs.typesafe.ai/api
Run: uv run uvicorn server:app --host 127.0.0.1 --port 8000
"""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("USE_TF", "0")
logger = logging.getLogger(__name__)
Name = Annotated[str, StringConstraints(strip_whitespace=False, min_length=1, pattern=r"\S")]
Description = str | dict[str, JsonValue] | list[JsonValue]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: Description


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    criteria: dict[Name, Description | None] = Field(min_length=1, max_length=255)


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    criteria: list[Description] = Field(min_length=2, max_length=10)


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    true: Description | None = None
    false: Description | None = None


class NoulQuestion(QuestionBase):
    type: Literal["noul"]
    criteria: NoulCriteria | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Description
    model: Name
    questions: dict[Name, Question] = Field(min_length=1)


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    probabilities: dict[str, Probability]
    confidence: Probability
    legend: dict[str, str]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: Probability


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    # Retain Laya's truncation diagnostics alongside the standard token counts.
    model_config = ConfigDict(extra="allow")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


@asynccontextmanager
async def lifespan(app: FastAPI):
    import laya

    configured_model = os.getenv("LAYA_MODEL", "models/korean-support-20000")
    local_path = ROOT / configured_model
    model_path = str(local_path) if local_path.exists() else configured_model
    app.state.model_name = os.getenv("LAYA_MODEL_NAME", "laya-korean-support")
    app.state.api_key = os.getenv("SYSTEM_ONE_API_KEY")
    app.state.inference_lock = Lock()
    app.state.agent = laya.load(model_path, device=os.getenv("LAYA_DEVICE", "cuda"))
    try:
        yield
    finally:
        app.state.agent.__exit__(None, None, None)
        app.state.agent = None


app = FastAPI(title="Laya System One API", version="0.1.0", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)


def authenticate(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
):
    key = request.app.state.api_key
    if key and (credentials is None or not secrets.compare_digest(
        credentials.credentials.encode("utf-8"), key.encode("utf-8")
    )):
        raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})


@app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(authenticate)])
def systemone(body: SystemOneRequest, request: Request):
    # Aliases route to the configured local model; never load a client-supplied path.
    model_name = request.app.state.model_name
    if body.model not in {model_name, "jev-latest", "jev-preview"}:
        raise HTTPException(422, f"Unknown model: {body.model}. Use {model_name} or jev-latest.")
    questions = {name: question.model_dump(exclude_none=True) for name, question in body.questions.items()}
    try:
        # The synchronous route runs in FastAPI's thread pool. Serialize GPU access.
        with request.app.state.inference_lock:
            result = request.app.state.agent.predict(body.state, questions)
        return SystemOneResponse(model=model_name, answers=result["answers"], usage=result["usage"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Laya inference failed")
        raise HTTPException(500, "Model inference failed") from exc


@app.get("/v1/models", dependencies=[Depends(authenticate)])
def models(request: Request):
    return {"models": [{"name": request.app.state.model_name, "description": "Local Laya model"}]}


@app.get("/health")
def health():
    return {"status": "ok"}
