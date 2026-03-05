---
language:
- ko
- en
license: apache-2.0
tags:
- kogum
- korean
- instruct
- sft
- chat
- causal-lm
- tool-calling
- thinking
library_name: transformers
pipeline_tag: text-generation
model-index:
- name: KoGum-0.5B-16k-Instruct
  results: []
---

# KoGum-0.5B-16k-Instruct

**KoGum**(코껌)은 **Ko**rean + **Gum**(껌)의 합성어입니다. 작은 모델에서 시작하여 크게 부풀어 오르는 풍선껌처럼, 점진적인 성장을 지향합니다.

**KoGum**은 LLM의 아키텍처 설계부터 pre-training, mid-training, SFT까지 전체 학습 파이프라인을 직접 구현하고 경험하기 위해 만든 한국어 특화 0.5B 소형 언어 모델입니다.
아키텍처 설계, 학습 데이터 구성, 학습 파이프라인 등에서 [KORMo](https://huggingface.co/KORMo-Team)를 많이 참고하였으며, KORMo에서 공개한 tokenizer와 데이터셋을 활용하였습니다.

> 이 모델은 **SFT (Supervised Fine-Tuning)** 단계의 instruction-tuned 모델로, 대화, 추론, 코드, 수학, tool calling을 지원합니다.
>
> **결론: 대 실패작입니다..!** 성능이 너무 안좋습니다. 하지만 이를 계기로 LLM의 전체 학습 파이프라인(아키텍처 설계 → pre-training → mid-training → SFT)을 직접 경험하고 이해할 수 있었으며, 이 경험을 바탕으로 더 발전된 모델을 만들어 나갈 예정입니다.

## Model Family

| Model | Stage | Description |
|---|---|---|
| [KoGum-0.5B-16k](https://huggingface.co/jiwon9703/KoGum-0.5B-16k) | Pre-train | Base model |
| [KoGum-0.5B-16k-mid](https://huggingface.co/jiwon9703/KoGum-0.5B-16k-mid) | Mid-train | Continued pre-training |
| **[KoGum-0.5B-16k-Instruct](https://huggingface.co/jiwon9703/KoGum-0.5B-16k-Instruct)** | SFT | Instruction-tuned chat model (this) |

## Architecture

LLaMA 스타일 아키텍처 기반 (GQA + RMSNorm + SwiGLU + RoPE).

| | |
|---|---|
| **Parameters** | ~581M |
| **Hidden Size** | 1024 |
| **Layers** | 24 |
| **Attention Heads** | 16 (GQA, 8 KV heads) |
| **Intermediate Size** | 5120 |
| **Vocab Size** | 125,041 |
| **Max Context** | 16,384 tokens |
| **RoPE Theta** | 500,000 |
| **Precision** | BFloat16 |

## Features

- **Chat**: 한국어/영어 대화
- **Thinking**: `<think>` 태그 기반 reasoning (enable/disable 가능)
- **Tool Calling**: `<tool_call>` / `<tool_response>` 기반 function calling
- **System Prompt**: 시스템 프롬프트 지원

## Training

### SFT Data

한국어 ~280K + 영어 ~120K = 총 ~400K samples (한/영 7:3):

**Korean (~280K)**

| Dataset | Samples | Description |
|---|---|---|
| [KORMo-Team/NemoPost-ko-synth-sft](https://huggingface.co/datasets/KORMo-Team/NemoPost-ko-synth-sft) | 200K | Korean QA |
| [KORMo-Team/IF-bilingual-sft](https://huggingface.co/datasets/KORMo-Team/IF-bilingual-sft) | 80K | Korean conversation (Magpie, MathInstruct, OpenHermes, etc.) |

**English (~120K)**

| Dataset | Samples | Description |
|---|---|---|
| [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) | 85K | Chat (30K), Code (20K), Math (20K), Tool Calling (15K) |
| [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | 35K | Diverse SFT (reasoning, multilingual, tool use, etc.) |

### Training Stats

| | |
|---|---|
| **Total Steps** | 8,000 |
| **Final Loss** | 1.81 |
| **Token Accuracy** | 60.9% |
| **Learning Rate** | 7e-6 → cosine decay |

## Usage

### Chat

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jiwon9703/KoGum-0.5B-16k-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "jiwon9703/KoGum-0.5B-16k-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "대한민국의 수도와 인구를 알려줘."}
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
input_ids = input_ids.to(model.device)

outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

### With System Prompt

```python
messages = [
    {"role": "system", "content": "당신은 친절한 한국어 AI 어시스턴트입니다."},
    {"role": "user", "content": "양자역학이 뭐야? 쉽게 설명해줘."}
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
input_ids = input_ids.to(model.device)

outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

### Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "서울 날씨 어때?"}
]

input_ids = tokenizer.apply_chat_template(
    messages, tools=tools, add_generation_prompt=True, return_tensors="pt"
)
input_ids = input_ids.to(model.device)

outputs = model.generate(input_ids, max_new_tokens=256)
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

## Chat Template

KoGum uses custom special tokens:
- `<|BOT|>`: Role boundary token
- `<|EOT|>`: End of turn
- `<think>...</think>`: Reasoning block
- `<tool_call>...</tool_call>`: Tool invocation
- `<tool_response>...</tool_response>`: Tool result

## Evaluation

> Benchmark 결과는 추후 업데이트 예정입니다.

## Limitations

- 0.5B 모델이므로 복잡한 추론이나 긴 문서 생성에는 한계가 있습니다.
- 학습 데이터에 포함되지 않은 최신 정보는 정확하지 않을 수 있습니다.
- Hallucination이 발생할 수 있으므로 사실 확인이 필요합니다.

## License

Apache 2.0