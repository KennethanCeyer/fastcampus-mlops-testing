from engine.engine import Engine, EngineStatus
from prompt import Intent, Slot


def test_engine_with_intent_close(engine_inst: Engine):
    # Given
    engine_inst.set_status(
        intent=Intent.INTENT_CLOSE, slot=Slot(slot_name="", slot_value="")
    )

    # When
    output_value = engine_inst.process()

    # Then
    assert output_value == "대화가 종료되었습니다."


def test_engine_with_intent_close_disposed(engine_inst: Engine):
    # Given
    engine_inst.set_status(
        intent=Intent.INTENT_CLOSE, slot=Slot(slot_name="", slot_value="")
    )
    engine_inst.dispose()

    # When
    output_value = engine_inst.process()

    # Then
    assert output_value == "이미 종료된 대화입니다."
