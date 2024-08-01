from utils import get_slot_value


def test_get_slot_value():
    # Given
    input_value = "결과: SLOT_ITEM"

    # When
    output_value = get_slot_value(input_value)

    # Then
    assert output_value == "SLOT_ITEM"
