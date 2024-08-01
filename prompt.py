from enum import StrEnum

from pydantic import BaseModel

from llm import generate_from_gemma
from utils import get_slot_value
from jinja2 import Environment, FileSystemLoader
from settings import PROJECT_ROOT


template_dir = PROJECT_ROOT / "prompts"
env = Environment(loader=FileSystemLoader(template_dir))


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


def get_intent(message: str) -> Intent:
    prompt = env.get_template("intent.txt").render({"intents": Intent.__doc__})
    response = generate_from_gemma(prompt + message)
    return response


def get_slot(message: str) -> Slot:
    prompt = env.get_template("slot.txt").render()
    response = generate_from_gemma(prompt + message)
    slot_name, slot_value = (response.split(",") + ["", ""])[:2]
    adjusted_slot_name = get_slot_value(slot_name)

    return Slot(
        slot_name=adjusted_slot_name,
        slot_value=slot_value,
    )


def get_answer_from_question(message: str) -> str:
    prompt = env.get_template("question.txt").render()
    response = generate_from_gemma(prompt + message)
    return response
