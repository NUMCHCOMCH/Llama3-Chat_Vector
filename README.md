#  Llama3-Chat_Vector-kor_Instruct : Chat-Vector를 활용한 한국어 LLAMA3 모델
- Chat-Vector Paper(https://arxiv.org/abs/2310.04799)
<p align="center" width="100%">
<img src="assert/ocelot.png" alt="NLP Logo" style="width: 50%;">
</p>

## Update Logs
- 2024.06.30: [🤗Llama3 모델 공개](cpm-ai/Ocelot-Ko-self-instruction-10.8B-v1.0)
이 모델은 Llama3-8B 모델의 Instruct 버전에 해당합니다.
---

### Reference Models:
1) meta-llama/Meta-Llama-3-8B(https://huggingface.co/meta-llama/Meta-Llama-3-8B)
2) meta-llama/Meta-Llama-3-8B-Instruct(https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
3) beomi/Llama-3-KoEn-8B(https://huggingface.co/beomi/Llama-3-KoEn-8B)

**Model Developers**: nebchi

## Model Information
* Chat_Vector는 학습된 가중치 매개변수를 더하고 빼는 것으로 사전 학습된 모델에 대화능력을 부여를 하는데, 이를 통해 영어 중심으로 학습된 LLAVA 모델에 대량의 한국어 코퍼스로 학습한 한국어 LLM의 언어 능력을 LLAVA에 주입하여 한국어 답변이 가능하도록 학습하였습니다.

### Description
* 이번 Llama3-Chat_Vector-kor_Instruct 모델은 대량의 한국어 코퍼스로 사전학습한 LLAMA 모델에서, 채팅 모델의 가중치를 더해 미세조정 없이 Instruction Model로 개발한 sLLM입니다.

### Inputs and outputs
*   **Input:** 질문, 프롬프트 또는 교정을 위한 문서와 같은 텍스트 문자열.
*   **Output:** 입력에 대한 응답으로 생성된 한국어 텍스트. 예를 들어, 질문에 대한 답변이나 이력서 평가.

---

#### 모델 실행 예시 코드 / multi gpu
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "nebchi/Llama3-Chat_Vector-kor",
)

model = AutoModelForCausalLM.from_pretrained(
    "nebchi/Llama3-Chat_Vector-kor",
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
streamer = TextStreamer(tokenizer)

messages = [
    {"role": "system", "content": "당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요."},
    {"role": "user", "content": "대한민국의 수도에 대해 알려줘"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=False,
    repetition_penalty=1.05,
    streamer = streamer
)
response = outputs[0][input_ids.shape[-1]:]
print(tok

### results
```python
대한민국의 수도는 서울특별시입니다.
서울특별시에는 청와대, 국회의사당, 대법원 등 대한민국의 주요 정부기관이 위치해 있습니다.
또한 서울시는 대한민국의 경제, 문화, 교육, 교통의 중심지로써 대한민국의 수도이자 대표 도시입니다.제가 도움이 되었길 바랍니다. 더 궁금한 점이 있으시면 언제든지 물어보세요!
```
---

**Citation**

```bibtex
@misc {Llama3-Chat_Vector-kor_Instruct,
	author       = { {nebchi} },
	title        = { Llama3-Chat_Vector-kor_Instruct },
	year         = 2024,
	url          = { https://huggingface.co/nebchi/Llama3-Chat_Vector-kor_llava },
	publisher    = { Hugging Face }
}
``'
}
```
