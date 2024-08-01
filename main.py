import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.float16},
    device="mps",
)

intents = {
    "INTENT_ORDER": "주문을 진행하려는 의도",
    "INTENT_ASK": "문의에 대한 의도",
    "INTENT_CLOSE": "문의 완료 및 대화 종료 의도",
}
intent_prompt = f"아래 사용자 대화의 의도를 INTENT 범위 내에서 어떤 것에 해당하는지 오직 분류된 결과만을 알려줘. 볼드 처리하지말아줘.\n선택가능한 의도 분류{intents.items()}\n유저 메시지\n:"
slot_prompt = f"아래 사용자 대화에서 의미있는 슬롯을 뽑아내어 인자로 활용할 수 있게 해줘. 키와 값을 콤마로 구분해. 설명 생략하고 결과만 알려줘, plaintext로 표현해.\n예시: 라면 2개 주문하고 싶은데요\n결과: SLOT_ITEM,2\n"
user_query = "생수 5개 주문할게요"
messages = [
    {"role": "user", "content": slot_prompt + user_query},
]

outputs = pipe(messages, max_new_tokens=1024)
assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
print(assistant_response)
