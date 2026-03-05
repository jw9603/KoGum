---
language:
- ko
- en
license: apache-2.0
tags:
- kogum
- korean
- pretrained
- causal-lm
- mid-training
library_name: transformers
pipeline_tag: text-generation
model-index:
- name: KoGum-0.5B-16k-mid
  results: []
---

# KoGum-0.5B-16k-mid

**KoGum**(코껌)은 **Ko**rean + **Gum**(껌)의 합성어입니다. 작은 모델에서 시작하여 크게 부풀어 오르는 풍선껌처럼, 점진적인 성장을 지향합니다.

**KoGum**은 LLM의 아키텍처 설계부터 pre-training, mid-training, SFT까지 전체 학습 파이프라인을 직접 구현하고 경험하기 위해 만든 한국어 특화 0.5B 소형 언어 모델입니다.
아키텍처 설계, 학습 데이터 구성, 학습 파이프라인 등에서 [KORMo](https://huggingface.co/KORMo-Team)를 많이 참고하였으며, KORMo에서 공개한 tokenizer와 데이터셋을 활용하였습니다.

> 이 모델은 **mid-training** 단계의 모델로, base 모델에 수학/코드/추론 능력을 강화하기 위해 continued pre-training을 수행한 결과입니다.
>
> **결론: 대 실패작입니다..!** 성능이 너무 안좋습니다. 하지만 이를 계기로 LLM의 전체 학습 파이프라인(아키텍처 설계 → pre-training → mid-training → SFT)을 직접 경험하고 이해할 수 있었으며, 이 경험을 바탕으로 더 발전된 모델을 만들어 나갈 예정입니다.

## Model Family

| Model | Stage | Description |
|---|---|---|
| [KoGum-0.5B-16k](https://huggingface.co/jiwon9703/KoGum-0.5B-16k) | Pre-train | Base model |
| **[KoGum-0.5B-16k-mid](https://huggingface.co/jiwon9703/KoGum-0.5B-16k-mid)** | Mid-train | Continued pre-training (this) |
| [KoGum-0.5B-16k-Instruct](https://huggingface.co/jiwon9703/KoGum-0.5B-16k-Instruct) | SFT | Instruction-tuned chat model |

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

## Training

### Goal

Base 모델의 **수학, 코드, 추론** 능력을 강화하기 위해 고품질 데이터로 continued pre-training을 수행했습니다.

### Data

11개 데이터셋을 개별 비율로 interleave하여 학습 (영어 65% + 한국어 35%):

**English (65%)**

| Dataset | Category | Ratio |
|---|---|---|
| [KORMo-Team/UltraFineWeb-filtered](https://huggingface.co/datasets/KORMo-Team/UltraFineWeb-filtered) | Web | 17% |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (cosmopedia-v2) | Web + Code + Textbook | 11% |
| [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (web_samples_v1) | Synthetic Textbook | 7% |
| [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) (finemath-3plus) | Math | 13% |
| [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) (CoT) | Math Reasoning | 10% |
| [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | Code Reasoning | 7% |

**Korean (35%)**

| Dataset | Category | Ratio |
|---|---|---|
| [KORMo-Team/UltraFineWeb-ko-synth](https://huggingface.co/datasets/KORMo-Team/UltraFineWeb-ko-synth) | Web | 11% |
| [KORMo-Team/FineWeb2-ko-synth](https://huggingface.co/datasets/KORMo-Team/FineWeb2-ko-synth) | Web | 8% |
| [KORMo-Team/NemoPost-ko-synth](https://huggingface.co/datasets/KORMo-Team/NemoPost-ko-synth) | Reasoning | 6% |
| [KORMo-Team/Cosmopedia-ko-synth](https://huggingface.co/datasets/KORMo-Team/Cosmopedia-ko-synth) | Textbook | 5% |
| [KORMo-Team/korean-public-corpus](https://huggingface.co/datasets/KORMo-Team/korean-public-corpus) | Public Data | 5% |

### Training Stats

| | |
|---|---|
| **Total Steps** | 1,500 |
| **Tokens Seen** | ~0.79B |
| **Final Loss** | 2.04 |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

tokenizer = AutoTokenizer.from_pretrained("jiwon9703/KoGum-0.5B-16k-mid", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "jiwon9703/KoGum-0.5B-16k-mid",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

input_text = "피타고라스 정리를 증명하면"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation

> Benchmark 결과는 추후 업데이트 예정입니다.

## License

Apache 2.0