from prompt import Intent, get_intent
from unittest.mock import patch


def test_get_intent_with_order_message():
    # Given
    input_value = "주문하고 싶은데요"
    with patch("prompt.generate_from_gemma") as mock_generate_from_gemma:
        mock_generate_from_gemma.return_value = Intent.INTENT_ORDER

        # When
        output_value = get_intent(input_value)

    # Then
    mock_generate_from_gemma.assert_called_once()
    assert mock_generate_from_gemma.call_args.args[0].endswith(input_value)
    assert output_value == Intent.INTENT_ORDER
