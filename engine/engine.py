from enum import IntEnum
from typing import Optional
from engine.actions import process_purchase
from prompt import Intent, Slot, get_answer_from_question, get_intent, get_slot


class EngineStatus(IntEnum):
    RUNNING = 0
    EXIT = 1


class Engine:

    def __init__(self):
        self._message: Optional[str] = None
        self._status: EngineStatus = EngineStatus.RUNNING
        self._intent: Optional[Intent] = None
        self._slot: Optional[Slot] = None

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, message: str) -> None:
        self._message = message
        self._intent = get_intent(message)
        self._slot = get_slot(message)

    def set_status(self, intent: Intent, slot: Slot) -> None:
        self._intent = intent
        self._slot = slot

    def do_ask(self) -> str:
        return get_answer_from_question(self.message)

    def do_order(self) -> str:
        process_purchase(self._slot)

        return f"{self._slot.slot_name}을 {self._slot.slot_value}로 처리했습니다."

    def do_close(self) -> str:
        self.dispose()
        return "대화가 종료되었습니다."

    def process(self) -> str:
        if self._status == EngineStatus.EXIT:
            return "이미 종료된 대화입니다."

        predicators = {
            Intent.INTENT_ASK: self.do_ask,
            Intent.INTENT_ORDER: self.do_order,
            Intent.INTENT_CLOSE: self.do_close,
        }
        action = predicators.get(self._intent)
        if action:
            return action()

        return "죄송합니다 잘 모르겠어요, 다시 설명해주시겠어요?"

    def dispose(self) -> None:
        self._status = EngineStatus.EXIT
