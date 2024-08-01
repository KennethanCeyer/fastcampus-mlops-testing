from transformers import pipeline
from utils import get_device
from transformers.pipelines.base import Pipeline
import torch
from functools import cache


@cache
def get_pipe() -> Pipeline:
    return pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.float16},
        device=get_device(),
    )


def generate_from_gemma(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
    ]
    pipe = get_pipe()
    outputs = pipe(messages, max_new_tokens=1024)
    return outputs[0]["generated_text"][-1]["content"].strip()
