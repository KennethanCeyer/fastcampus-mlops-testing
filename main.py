import json
import torch
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class ChatbotMessageRequest(BaseModel):
    message: str


class ChatbotMessageResponse(BaseModel):
    message: str


pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.float16},
    device="mps",
)


app = FastAPI()
intents = {
    "INTENT_ORDER": "주문을 진행하려는 의도",
    "INTENT_ASK": "문의에 대한 의도",
    "INTENT_CLOSE": "문의 완료 및 대화 종료 의도",
}
intent_prompt = f"""
아래 사용자 대화의 의도를 INTENT 범위 내에서
어떤 것에 해당하는지 오직 분류된 결과만을 알려줘.

볼드 처리하지말아줘.
선택가능한 의도 분류 {intents.items()}

유저 메시지:
"""
slot_prompt = """
아래 사용자 대화에서 의미있는 슬롯을 뽑아내어
인자로 활용할 수 있게 해줘.

키와 값을 콤마로 구분해.
스니펫으로 처리하지 말아줘.

설명 생략하고 결과만 알려줘.

예시: 라면 2개 주문하고 싶은데요
결과: SLOT_ITEM,2

"""


@app.post("/chatbot/message", response_model=ChatbotMessageResponse)
async def send_message(request: ChatbotMessageRequest) -> ChatbotMessageResponse:
    messages = [
        {"role": "user", "content": slot_prompt + request.message},
    ]
    outputs = pipe(messages, max_new_tokens=1024)

    response = outputs[0]["generated_text"][-1]["content"].strip()
    slot_name, slot_value = response.split(",")[:2]

    slot_json = {"slot_name": slot_name, "slot_value": int(slot_value)}
    return ChatbotMessageResponse(message=json.dumps(slot_json))
