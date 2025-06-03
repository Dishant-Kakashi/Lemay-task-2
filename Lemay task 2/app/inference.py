
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from dotenv import load_dotenv
load_dotenv()

# Load model name and token from environment variables
model_name = os.getenv("MODEL_NAME")
hf_token = os.getenv("HF_TOKEN")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=hf_token)

# Create the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def answer_question(question: str, context: str) -> str:
    QA_input = {
        "question": question,
        "context": context
    }
    result = qa_pipeline(QA_input)
    return result['answer']

