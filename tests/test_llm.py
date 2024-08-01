from llm import generate_from_gemma, get_pipe
from unittest import mock


def test_generate_from_gemma():
    # Given
    input_value = "hello world"

    # When
    with mock.patch("llm.get_pipe") as mock_get_pipe:
        mock_pipe = mock.Mock()
        mock_pipe.return_value = [{"generated_text": [{"content": "foobar"}]}]
        mock_get_pipe.return_value = mock_pipe

        output_value = generate_from_gemma(input_value)

        mock_get_pipe.assert_called_once()
        mock_pipe.assert_called_once_with(
            [
                {"role": "user", "content": input_value},
            ],
            max_new_tokens=1024,
        )

    # Then
    assert output_value == "foobar"
