import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import answer_question

app = FastAPI()

# Pydantic model for request
class QARequest(BaseModel):
    question: str
    context: str

# Health check
@app.get("/")
def read_root():
    return {"message": "QA API is running", "worker_pid": os.getpid()}

# Question Answering endpoint
@app.post("/generate")
async def generate_answer(payload: QARequest):
    result = answer_question(payload.question, payload.context)
    return {
        "response": result,
        "worker_pid": os.getpid()
    }