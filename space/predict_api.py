"""FastAPI entry point for the fine-tuned Qwen3 passage QA model."""

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from qwen_inference import (
    MAX_CONTEXT_CHARACTERS,
    MAX_QUESTION_CHARACTERS,
    QWEN3_MODEL_NAME,
    QuestionTooLongError,
    generative_answer,
    load_qwen3,
)

generation_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_qwen3()
    yield


app = FastAPI(title="Transformer QA Inference", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARACTERS)
    context: str = Field(min_length=1, max_length=MAX_CONTEXT_CHARACTERS)


class PredictResponse(BaseModel):
    answer: str
    model: str


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        with generation_lock:
            answer, _ = generative_answer(request.context, request.question)
    except QuestionTooLongError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(answer=answer, model=QWEN3_MODEL_NAME)
