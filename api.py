from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import ask_assistant


app = FastAPI(title="Internal AI Knowledge Assistant API")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"message": "Internal AI Knowledge Assistant API is running."}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        answer = ask_assistant(request.question)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))