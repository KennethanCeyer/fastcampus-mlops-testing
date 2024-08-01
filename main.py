from fastapi import FastAPI
from engine.engine import Engine
from model import ChatbotMessageRequest, ChatbotMessageResponse
from prompt import get_intent, get_slot


app = FastAPI()
engine = Engine()


@app.post("/chatbot/message", response_model=ChatbotMessageResponse)
async def send_message(request: ChatbotMessageRequest) -> ChatbotMessageResponse:
    engine.message = request.message
    response = engine.process()
    return ChatbotMessageResponse(message=response)
