# KoGum Pre-training Guide

KoGum 0.5B 모델의 사전학습 가이드입니다.

## 학습 설정

### 모델 사양
- **파라미터**: 616.9M (0.62B)
- **아키텍처**: Llama-style decoder-only
- **컨텍스트 길이**: 4096 tokens
- **어텐션**: GQA (2:1 ratio, 16 query heads, 8 KV heads)
- **활성화 함수**: SwiGLU (5× expansion)
- **정규화**: RMSNorm
- **레이어**: 24 layers, hidden_size 1024

### 토크나이저
- **Vocab size**: 80,038 (80,000 base + 38 special tokens)
- **알고리즘**: BPE (Byte Pair Encoding)
- **학습 데이터**: 7M samples (70% Korean, 30% English)
- **Special tokens**: `<|PAD|>`, `<|BOS|>`, `<|EOS|>`, `<|UNK|>` 외 34개

### 학습 데이터
- **데이터셋**:
  - Korean: KORMo-Team/korean-web-collection
  - English: KORMo-Team/dclm-baseline-filtered
  - Interleaved (all data from both datasets)
- **모드**: Streaming (메모리 효율적)
- **시퀀스 길이**: 4096 tokens (sequence packing 적용)
- **Validation**: 1,000 samples

## 학습 실행

### 1. Full-scale Pre-training (단일 GPU)

```bash
# 20B 토큰 학습 (Chinchilla optimal)
bash scripts/run_pretrain.sh
```

**설정**:
- Target tokens: 20B (Chinchilla optimal: 20× parameters)
- Max steps: 4,768
- Per-device batch size: 8
- Gradient accumulation: 자동 계산
- Effective batch size: 1024 sequences (~4M tokens per step)
- Learning rate: 3e-4
- Warmup: 3% of steps
- LR scheduler: Cosine
- Precision: BF16
- Gradient checkpointing: Enabled
- Save frequency: 500 steps
- Max checkpoints: 5

**예상 소요 시간** (단일 A100 80GB 기준):
- 단일 GPU는 배치 크기 제약으로 비효율적
- **멀티 GPU 사용 강력 권장**

### 2. Full-scale Pre-training (멀티 GPU) - 권장

```bash
# 분산 학습 (torchrun 사용)
bash scripts/run_pretrain_distributed.sh
```

**설정**:
- Context length: 4096 tokens
- Per-device batch size: 8
- Total batch size: 1024 sequences (~4M tokens/step)
- Max steps: 4,768
- GPU 수에 따라 gradient accumulation 자동 조정
- Streaming mode with spike detection

**예상 소요 시간** (A100 80GB 기준):
- **4x A100**: ~27s/step → ~36 hours (1.5일)
- 자동 체크포인트 재개 기능 포함
- 스파이크 감지 및 배치 저장 자동화

## 하이퍼파라미터 상세

### 배치 크기
```
Effective batch size = Per-device batch × Gradient accumulation × Num GPUs
                     = 8 × 32 × 4
                     = 1024 sequences
                     = 1024 × 4096 tokens
                     ≈ 4.2M tokens per step
```

### Learning Rate Schedule
```
Initial LR: 0
Warmup: 0 → 3e-4 (3% of steps, ~143 steps)
Main: 3e-4 → 0 (Cosine decay)
Total steps: 4,768
```

### Weight Decay
- **적용**: Linear layer weights
- **제외**: LayerNorm, RMSNorm, Embedding, Bias
- **값**: 0.1

### Optimizer
- **알고리즘**: AdamW
- **Beta1**: 0.9
- **Beta2**: 0.95
- **Epsilon**: 1e-8
- **Max grad norm**: 1.0

## 모니터링

### WandB
학습 중 다음 메트릭이 자동으로 로깅됩니다:
- Loss (train)
- Learning rate
- Gradient norm
- Tokens per second
- GPU memory usage
- Step time

WandB 프로젝트: https://wandb.ai/jwjw9603-no/kogum-pretrain

### 체크포인트
체크포인트는 `./kogum-pretrain/` 디렉토리에 저장됩니다:
```
kogum-pretrain/
├── checkpoint-1000/
├── checkpoint-2000/
├── ...
└── final_model/
```

각 체크포인트 크기: ~2.3GB

## 학습 재개

```bash
# 마지막 체크포인트에서 재개
bash scripts/run_pretrain.sh

# 특정 체크포인트에서 재개
/home/infidea/anaconda3/envs/kormo/bin/python scripts/train.py \
    --resume_from_checkpoint ./kogum-pretrain/checkpoint-50000 \
    [다른 인자들...]
```

## 트러블슈팅

### OOM (Out of Memory)
```bash
# Per-device batch size 줄이기
PER_DEVICE_BATCH_SIZE=2  # 4 → 2

# Gradient accumulation 증가 (배치 크기 유지)
GRAD_ACCUM=$((TOTAL_BATCH_SIZE / NUM_GPUS / PER_DEVICE_BATCH_SIZE))
```

### 느린 학습 속도
1. Gradient checkpointing 비활성화 (메모리 여유 있을 때)
2. Mixed precision (BF16/FP16) 확인
3. DataLoader workers 수 조정 (`--preprocessing_num_workers`)

### WandB 연결 실패
```bash
# .env 파일 확인
cat .env

# WandB 재로그인
wandb login --relogin
```

## 참고 자료

### 유사 모델 학습 설정
- **Llama 2 7B**: 2T tokens, batch 4M tokens
- **Gemma 2B**: 2T tokens, batch 512K tokens
- **Qwen 0.5B**: 3T tokens, batch 4M tokens
- **KoGum 0.5B**: 20B tokens, batch 4M tokens (Chinchilla optimal)

### Chinchilla Optimal 계산
```
Optimal tokens ≈ 20 × Parameters
For 0.5B model: 20 × 0.5B = 10B tokens (minimum)
Current setting: 20B tokens (2× optimal, for better performance)
```

### 향후 실험 옵션
- Total tokens: 20B → 50B (추가 성능 향상)
- Learning rate: 3e-4 ~ 5e-4 범위 실험
- Batch size: 4M 유지 (검증됨)
