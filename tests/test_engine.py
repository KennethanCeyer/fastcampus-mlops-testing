from main import Engine, EngineStatus, Intent, Slot


def test_engine_with_intent_close():
    # Given
    engine = Engine()
    engine.set_status(
        intent=Intent.INTENT_CLOSE, slot=Slot(slot_name="", slot_value="")
    )

    # When
    output_value = engine.process()

    assert output_value == "대화가 종료되었습니다."


def test_engine_with_intent_close_disposed():
    # Given
    engine = Engine()
    engine.set_status(
        intent=Intent.INTENT_CLOSE, slot=Slot(slot_name="", slot_value="")
    )
    engine.set_status = EngineStatus.EXIT

    # When
    output_value = engine.process()

    assert output_value == "대화가 종료되었습니다."
