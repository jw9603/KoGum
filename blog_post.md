# KoGum: 0.5B 한국어 LLM을 처음부터 끝까지 만들어본 이야기

## 들어가며

최근 LLM이 급속도로 발전하면서 모델을 "사용"하는 것은 쉬워졌지만, **직접 만들어보는 경험**은 여전히 드물다. 나는 LLM의 전체 파이프라인을 직접 경험해보고 싶었다. 아키텍처 설계, 모델 구현, pre-training, mid-training, SFT까지 — 논문에서 읽기만 했던 것들을 실제로 부딪혀보면서 이해하고 싶었다.

그 결과물이 **KoGum**(코껌)이다. Korean + Gum(껌)의 합성어로, 작은 모델에서 시작하여 크게 부풀어 오르는 풍선껌처럼 점진적인 성장을 지향한다는 의미를 담았다. 0.5B 파라미터의 한국어 특화 소형 언어 모델이다.

> 결론부터 말하면 **대 실패작**이다. 하지만 이 실패 과정에서 배운 것들이 매우 많았다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| **모델명** | KoGum-0.5B-16k |
| **파라미터** | ~581M |
| **아키텍처** | LLaMA-style (GQA + RMSNorm + SwiGLU + RoPE) |
| **Context Length** | 16,384 tokens (16K) |
| **Vocab Size** | 125,041 (한국어 최적화 BPE) |
| **학습 단계** | Pre-training → Mid-training → SFT |
| **학습 환경** | Azure NCv4 (A100 80GB × 4) |
| **참고** | [KORMo](https://huggingface.co/KORMo-Team) (tokenizer, 데이터셋, 학습 파이프라인 참고) |

---

## 1. 아키텍처 설계 및 구현

### LLaMA-style Architecture

최근 대부분의 LLM이 채택하는 LLaMA 스타일 아키텍처를 기반으로 설계했다. 핵심 구성 요소는 다음과 같다:

```
Input → Embedding → [DecoderLayer × 24] → RMSNorm → LM Head → Output

DecoderLayer:
  x → RMSNorm → GQA Attention → +residual →
      RMSNorm → SwiGLU MLP → +residual
```

#### Grouped Query Attention (GQA)

- Query 16 heads, Key-Value 8 heads (2:1 비율)
- MHA 대비 메모리 사용량 절감하면서 MQA보다 나은 성능
- KV 캐시 크기가 절반으로 줄어 추론 속도 향상

#### RMSNorm (Pre-LN)

```python
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

LayerNorm과 달리 평균(mean)을 빼지 않아 연산이 더 가벼우면서 LLM에서는 성능 차이가 없다. Pre-LN 구조로 각 sublayer 이전에 normalization을 적용하여 학습 안정성을 확보했다.

#### SwiGLU Feed-Forward Network

```python
output = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
# 1024 → 5120 → 1024 (5× expansion)
```

기존 FFN의 ReLU를 SiLU(Swish) 기반의 gating mechanism으로 대체한 구조다. 파라미터는 조금 더 많아지지만 같은 크기 대비 성능이 더 좋다.

#### RoPE (Rotary Position Embedding)

- θ = 500,000 (긴 문맥 지원을 위한 높은 base frequency)
- 16K context를 안정적으로 처리
- 학습 가능한 파라미터 없이 위치 정보를 인코딩

### 구현 과정에서의 이슈들

**16K Context에서의 Attention Overflow 문제**

SDPA(Scaled Dot-Product Attention)의 Math 백엔드는 B×H×L×L 크기의 attention matrix를 메모리에 직접 할당한다. 16K context에서는 이 크기가 int32 범위(2^31-1)를 초과하여 overflow가 발생한다. FlashAttention은 attention matrix를 메모리에 올리지 않으므로 이 문제를 회피할 수 있다.

```python
enable_math_backend = (seq_length <= 8192)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=enable_math_backend,  # 8K 이하에서만 Math 백엔드 허용
    enable_mem_efficient=False
):
    attn_output = F.scaled_dot_product_attention(...)
```

**BFloat16 정밀도 관리**

RMSNorm 내부에서 variance 계산 시 BFloat16의 정밀도 한계로 인해 수치 불안정이 발생할 수 있다. 입력을 FP32로 변환한 후 계산하고, 결과를 다시 원래 dtype으로 되돌리는 방식으로 처리했다.

**Attention Mask dtype 불일치**

추론 시 `torch_dtype="auto"`를 사용하면 모델 가중치와 attention mask의 dtype이 불일치하여 SDPA에서 에러가 발생하는 문제가 있었다. attention mask의 dtype을 query 텐서의 dtype에 맞춰 변환하는 코드를 추가하여 해결했다.

---

## 2. Pre-training: 20B 토큰으로 기초 학습

### 데이터 구성

한국어 70% + 영어 30% 비율로 혼합 학습했다:

| Dataset | Language | Ratio |
|---|---|---|
| KORMo-Team/korean-web-collection | Korean | 70% |
| KORMo-Team/dclm-baseline-filtered | English | 30% |

### Sequence Packing

16K context를 효율적으로 활용하기 위해 여러 문서를 하나의 시퀀스에 연결하는 packing 방식을 사용했다. 이렇게 하면 짧은 문서에서 발생하는 padding waste를 제거하여 GPU utilization을 극대화할 수 있다.

각 문서 사이에 EOS 토큰을 삽입하여 문서 경계를 표시하고, document_ids를 추적하여 cross-document attention을 방지하는 intra-document masking도 구현했다. 다만 gradient checkpointing과의 호환성 문제로 intra-document masking은 비활성화한 채 학습을 진행했다.

### 학습 설정

| 항목 | 값 |
|---|---|
| Total Tokens | 20B (Chinchilla optimal: 파라미터의 ~20배) |
| Total Steps | 4,768 |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| Learning Rate | 3e-4 (cosine decay with 3% warmup) |
| Weight Decay | 0.1 (Linear weights만 적용) |
| Max Grad Norm | 1.0 |
| Precision | BFloat16 mixed precision |
| Gradient Checkpointing | Enabled (메모리 절약) |

### Selective Weight Decay

모든 파라미터에 동일한 weight decay를 적용하는 것이 아니라, **Linear layer의 weight에만** decay를 적용했다. RMSNorm, Embedding, Bias 파라미터는 decay에서 제외했는데, 이들은 특성이 다른 파라미터이므로 decay가 오히려 학습을 방해할 수 있기 때문이다.

### Spike Detection

학습 중 loss spike나 gradient norm 이상치를 자동으로 감지하는 `BatchSpikeDetector`를 구현했다. 이동 평균의 2배를 임계값으로 설정하고, 연속 2 step 이상 이상치가 관측되면 해당 batch의 실제 데이터 샘플을 저장하여 사후 분석이 가능하도록 했다.

### 발생했던 문제: SIGSEGV

Pre-training 초기에 학습 도중 **SIGSEGV (Segmentation Fault)** 가 발생하는 문제가 있었다. 여러 가지 원인을 조사한 결과, NCCL의 GPU topology 인식 문제였다. Azure NCv4의 기본 NCCL topology XML은 GPU 0-3을 모두 참조하는데, `CUDA_VISIBLE_DEVICES=2,3`으로 특정 GPU만 사용할 경우 GPU 0,1이 없어서 크래시가 발생했다. GPU 2,3만 포함하는 커스텀 NCCL XML 파일을 생성하여 해결했다.

---

## 3. Mid-training: 수학·코드·추론 능력 강화

### 목적

Base 모델의 수학, 코드, 추론 능력을 집중적으로 강화하기 위해 고품질 데이터로 continued pre-training을 수행했다.

### 데이터 구성

11개 데이터셋을 개별 비율로 interleave하여 학습했다 (영어 65% + 한국어 35%):

**English (65%)**

| Dataset | Category | Ratio |
|---|---|---|
| UltraFineWeb-filtered | Web | 17% |
| smollm-corpus/cosmopedia-v2 | Web + Code + Textbook | 11% |
| cosmopedia/web_samples_v1 | Synthetic Textbook | 7% |
| finemath/finemath-3plus | Math | 13% |
| OpenMathReasoning (CoT) | Math Reasoning | 10% |
| OpenCodeReasoning | Code Reasoning | 7% |

**Korean (35%)**

| Dataset | Category | Ratio |
|---|---|---|
| UltraFineWeb-ko-synth | Web | 11% |
| FineWeb2-ko-synth | Web | 8% |
| NemoPost-ko-synth | Reasoning | 6% |
| Cosmopedia-ko-synth | Textbook | 5% |
| korean-public-corpus | Public Data | 5% |

### 학습 결과

| 항목 | 값 |
|---|---|
| Total Steps | 1,500 |
| Tokens Seen | ~0.79B |
| Final Loss | 2.04 |

Mid-training에서는 pre-training 대비 영어 비율을 65%로 높였다. 수학, 코드 관련 고품질 데이터가 대부분 영어로 되어 있기 때문이다.

---

## 4. SFT: Instruction Tuning

### Chat Template 설계

SFT를 위해 다양한 special token을 설계했다:

```
<|BOS|>      - 시퀀스 시작
<|BOT|>      - 어시스턴트 턴 시작
<|EOT|>      - 턴 종료
<think>      - 추론 블록 시작
</think>     - 추론 블록 종료
<tool_call>  - 함수 호출
<tool_response> - 함수 결과
```

이를 통해 일반 대화, reasoning (chain-of-thought), tool calling을 하나의 통합된 chat template으로 지원할 수 있도록 했다.

### 데이터 구성

한국어 ~280K + 영어 ~120K = 총 ~400K samples:

**Korean (~280K)**
- NemoPost-ko-synth-sft: 200K (QA 형식)
- IF-bilingual-sft (Korean subset): 80K (대화 형식)

**English (~120K)**
- Nemotron-Post-Training-Dataset-v1: 85K (Chat 30K, Code 20K, Math 20K, Tool Calling 15K)
- smoltalk2/SFT: 35K (think/no_think reasoning 포함)

### Label Masking

SFT에서는 **어시스턴트의 응답 부분만** loss 계산에 포함시키고, 사용자/시스템 프롬프트는 `labels = -100`으로 마스킹했다. `<think>`, `<tool_call>` 등의 내용도 어시스턴트 응답에 포함되므로 학습 대상이 된다.

### 학습 결과

| 항목 | 값 |
|---|---|
| Total Steps | 8,000 (1 epoch 기준 best) |
| Final Loss | 1.81 |
| Token Accuracy | 60.9% |
| Learning Rate | 7e-6 → cosine decay |

### 발생했던 문제: LR Scheduler Reset

2 epoch 학습을 진행했는데, epoch 전환 시 learning rate scheduler가 리셋되는 문제가 있었다. 이미 감소한 learning rate가 다시 초기값으로 돌아가면서 학습이 불안정해졌고, 결국 1 epoch에서의 best checkpoint (step 8000)를 최종 모델로 선택했다.

---

## 5. 기술적 구현 세부사항

### Custom Trainer

HuggingFace Trainer를 상속한 `KoGumTrainer`를 구현했다. 주요 커스터마이징:

- **Selective Weight Decay**: Linear weight에만 decay, Norm/Embedding/Bias 제외
- **Token-level Accuracy**: Forward pass에서 logits를 별도로 저장하지 않고 accuracy 계산 (125K vocab × 16K context의 logits은 4GB+ 메모리 소모)
- **8-bit Optimizer 지원**: bitsandbytes를 활용한 8-bit AdamW, Embedding은 FP32로 유지
- **Spike Detection 통합**: 학습 루프에 이상치 감지 기능 내장

### 메모리 최적화

0.5B 모델이지만 16K context에서는 메모리 관리가 중요했다:

- **Gradient Checkpointing**: Activation을 저장하지 않고 backward 시 재계산
- **Logits 미반환**: Training 시 logits를 output에 포함하지 않아 메모리 절약
- **FlashAttention**: Attention matrix를 메모리에 올리지 않아 O(N²) → O(N) 메모리

### 프로젝트 구조

```
KoGum/
├── src/kogum/
│   ├── model/
│   │   ├── modeling_kogum.py      # 모델 아키텍처 전체 구현
│   │   └── configuration_kogum.py  # 모델 설정
│   ├── train/
│   │   ├── pretrain.py            # Pre-training 스크립트
│   │   ├── midtrain.py            # Mid-training 스크립트
│   │   ├── sft.py                 # SFT 스크립트
│   │   ├── trainer.py             # Custom HF Trainer
│   │   └── spike_detector.py      # 학습 이상치 감지
│   └── data_utils/
│       ├── collator.py            # Data Collator
│       └── packing.py             # Sequence Packing
├── kogum-tokenizer/                # BPE Tokenizer (125K vocab)
└── scripts/                        # 분산 학습 셸 스크립트
```

---

## 6. 결과 및 회고

### 솔직한 평가: 대 실패작

결론부터 말하면, KoGum은 실용적인 수준의 모델이 아니다. 간단한 수학 문제(3x+5=20)도 틀리고, reasoning mode에서는 `<think>` 블록에서 반복 루프에 빠지며, hallucination이 심하다.

0.5B이라는 작은 크기도 원인이지만, 학습 과정의 여러 결정들이 최적이 아니었던 것도 크다:

- **Intra-document masking 비활성화**: Gradient checkpointing과의 충돌로 문서 간 attention 방지를 끌 수밖에 없었는데, 이것이 모델 품질에 부정적 영향을 미쳤을 수 있다
- **Mid-training 데이터량**: 0.79B 토큰은 능력 강화에 충분하지 않았을 수 있다
- **LR Scheduler Reset**: 2 epoch SFT에서 scheduler가 리셋되어 1 epoch만 활용
- **하이퍼파라미터 튜닝 부족**: 컴퓨팅 자원의 제약으로 충분한 실험을 하지 못했다

### 배운 것들

실패작이지만, 이 프로젝트를 통해 얻은 것은 많다:

1. **아키텍처 이해**: GQA, RMSNorm, SwiGLU, RoPE 등의 구성 요소를 직접 구현하면서 각 컴포넌트의 역할과 설계 이유를 체감할 수 있었다

2. **학습 파이프라인 경험**: Pre-training → Mid-training → SFT의 3단계 학습이 각각 어떤 역할을 하는지, 데이터 구성과 하이퍼파라미터가 어떻게 달라지는지 직접 경험했다

3. **디버깅 능력**: SIGSEGV, NCCL topology, dtype mismatch, memory overflow 등 GPU 기반 분산 학습에서 발생하는 다양한 문제들을 해결하면서 실전 디버깅 능력을 키웠다

4. **메모리 관리**: 16K context에서의 메모리 최적화 기법들 (gradient checkpointing, FlashAttention, logits 관리)을 직접 적용해봤다

5. **데이터 엔지니어링**: Sequence packing, document boundary handling, streaming dataset, interleaved sampling 등 대규모 학습을 위한 데이터 처리 파이프라인을 구축했다

6. **Spike Detection**: 학습 안정성을 모니터링하고 이상치를 자동 감지하는 시스템을 구현하여, 문제 발생 시 원인 분석이 가능하도록 했다

### 앞으로의 계획

이번 경험을 바탕으로 더 발전된 모델을 만들어 나갈 예정이다:

- 더 많은 학습 데이터와 더 나은 데이터 품질 관리
- Intra-document masking 문제 해결
- 체계적인 하이퍼파라미터 튜닝
- 벤치마크 기반의 체계적인 평가

---

## 모델 링크

- [KoGum-0.5B-16k](https://huggingface.co/jiwon9703/KoGum-0.5B-16k) — Pre-trained base model
- [KoGum-0.5B-16k-mid](https://huggingface.co/jiwon9703/KoGum-0.5B-16k-mid) — Mid-trained (math/code/reasoning 강화)
- [KoGum-0.5B-16k-Instruct](https://huggingface.co/jiwon9703/KoGum-0.5B-16k-Instruct) — SFT instruction-tuned
