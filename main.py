from enum import Enum, EnumMeta, IntEnum, StrEnum
import json
import torch
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import re
from typing import Optional


class ChatbotMessageRequest(BaseModel):
    message: str


class ChatbotMessageResponse(BaseModel):
    message: str


class Intent(StrEnum):
    """
    INTENT_ORDER: Intent of ordering a product
    INTENT_ASK: Intent of asking some question
    INTENT_CLOSE: Intent of willing to close the conversation
    """

    INTENT_ORDER = "INTENT_ORDER"
    INTENT_ASK = "INTENT_ASK"
    INTENT_CLOSE = "INTENT_CLOSE"


class Slot(BaseModel):
    slot_name: str
    slot_value: str


class EngineStatus(IntEnum):
    RUNNING = 0
    EXIT = 1


class Engine:
    def __init__(self):
        self.status: EngineStatus = EngineStatus.RUNNING
        self.message: Optional[str] = None
        self.intent: Optional[Intent] = None
        self.slot: Optional[Slot] = None

    def set_message(self, message: str) -> None:
        self.message = message

    def set_status(self, intent: Intent, slot: Slot) -> None:
        self.intent = intent
        self.slot = slot

    def do_ask(self) -> str:
        return get_answer_from_question(self.message)

    def do_order(self) -> str:
        process_purchase(self.slot)

        return f"{self.slot.slot_name}을 {self.slot.slot_value}로 처리했습니다."

    def do_close(self) -> str:
        self.dispose()
        return "대화가 종료되었습니다."

    def process(self) -> str:
        if self.status == EngineStatus.EXIT:
            return "이미 종료된 대화입니다."

        if self.intent == Intent.INTENT_ASK:
            return self.do_ask()
        elif self.intent == Intent.INTENT_ORDER:
            return self.do_order()
        elif self.intent == Intent.INTENT_CLOSE:
            return self.do_close()

        return "죄송합니다 잘 모르겠어요, 다시 설명해주시겠어요?"

    def dispose(self) -> None:
        self.status = EngineStatus.EXIT


pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.float16},
    device="mps",
)


app = FastAPI()
engine = Engine()


def get_slot_value(slot_value: str) -> str:
    groups = re.search(r"(SLOT_\w+)", slot_value)

    if not groups:
        return ""

    return groups.group(1)


def generate_from_gemma(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(messages, max_new_tokens=1024)
    return outputs[0]["generated_text"][-1]["content"].strip()


def get_intent(message: str) -> Intent:
    prompt = f"""
    아래 사용자 대화의 의도를 INTENT 범위 내에서
    어떤 것에 해당하는지 오직 분류된 결과만을 알려줘.

    볼드 처리하지말아줘.
    선택가능한 의도 분류 {Intent.__doc__}

    유저 메시지:
    """

    response = generate_from_gemma(prompt + message)
    return response


def get_slot(message: str) -> Slot:
    prompt = """
    아래 사용자 대화에서 의미있는 슬롯을 뽑아내어
    인자로 활용할 수 있게 해줘.

    키와 값을 콤마로 구분해.
    스니펫으로 처리하지 말아줘.

    설명 생략하고 결과만 알려줘.

    예시: 라면 2개 주문하고 싶은데요
    결과: SLOT_ITEM,2

    """
    response = generate_from_gemma(prompt + message)
    slot_name, slot_value = (response.split(",") + ["", ""])[:2]
    adjusted_slot_name = get_slot_value(slot_name)

    return Slot(
        slot_name=adjusted_slot_name,
        slot_value=slot_value,
    )


def get_answer_from_question(message: str) -> str:
    prompt = """
    Answer the following question

    QUESTION:
    """
    response = generate_from_gemma(prompt + message)
    return response


def process_purchase(slot: Slot) -> None:
    print(f"purchasing process with {slot=}")


@app.post("/chatbot/message", response_model=ChatbotMessageResponse)
async def send_message(request: ChatbotMessageRequest) -> ChatbotMessageResponse:
    engine.set_message(request.message)
    intent = get_intent(request.message)
    slot = get_slot(request.message)
    engine.set_status(intent, slot)
    response = engine.process()
    print(intent, slot)

    return ChatbotMessageResponse(message=response)
