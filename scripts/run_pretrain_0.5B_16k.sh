#!/bin/bash
# KoGum 0.5B (16K context) Pre-training Script (Distributed)
# 목적: KORMo tokenizer로 0.5B 모델 학습 (16K context, gradient checkpointing)

# 어떤 명령이든 실패하면 즉시 스크립트 종료, 없으면 에러가 나도 다음줄 계속 실행됨
set -e 

# kogum 가상환경 명시적 활성화
source ~/.bashrc
conda activate kogum

# Python 경로 확인
PYTHON_PATH=$(which python)             # 활성화된 환경의 python 경로 확인
echo "Using Python: $PYTHON_PATH"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# .env 파일 로드
if [ -f .env ]; then                    # 현재 디렉토리에 .env라는 파일이 존재하는지 확인, 존재하면 true
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs) # .env 파일에서 주석(#)이 아닌 라인들을 읽어서 환경 변수로 설정 (예: export VAR=value)
fi
#  grep -v '^#'  → 주석(#) 제외
#  xargs         → 한 줄로 합침
#  export        → 환경변수로 등록

# PyTorch CUDA 메모리 설정 (16K context 최적화)
# Disable expandable_segments to avoid memory fragmentation issues in DDP
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# PyTorch는 GPU 메모리를 cudaMalloc/cudaFree로 매번 할당/해제하지 않고, CUDACaching/Allocator라는 캐시 allocator가 큰 블록을 잡아두고 쪼개서 재사용함.
# max_split_size_mb 설정은 이 캐시 allocator가 너무 작은 조각들로 메모리를 쪼개지 않도록 방지함. (메모리 단편화 완화)


# NCCL (NVIDIA Collective Communications Library) = GPU 간 통신 라이브러리. 설정 (GPU 2,3 사용 시 XML 문제 해결)
export NCCL_TOPO_FILE=/tmp/custom_nccl_topo.xml
export NCCL_GRAPH_FILE=/tmp/custom_nccl_graph.xml
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "NCCL_TOPO_FILE=$NCCL_TOPO_FILE"
echo "NCCL_GRAPH_FILE=$NCCL_GRAPH_FILE"

echo "========================================"
echo "KoGum 0.5B (16K) Pre-training"
echo "========================================"
echo "Model: 0.5B parameters"
echo "Context: 16,384 tokens"
echo "Tokenizer: KORMo (125K vocab)"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# =============================================================================
# 하드웨어 설정
# =============================================================================
# CUDA_VISIBLE_DEVICES 기반으로 GPU 개수 계산 (안전)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "❌ CUDA_VISIBLE_DEVICES is not set. Please set it in .env file."
    exit 1
fi

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPUs: $NUM_GPUS"

if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "❌ Invalid CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (e.g., 0,1)"
  exit 1
fi

# =============================================================================
# 배치 크기 설정 (16K context, 0.5B model)
# =============================================================================
# Context length: 16384 tokens
# Model: 0.5B parameters (smaller than 1B)
# Gradient checkpointing: enabled (for 16K context)
# Memory: ~52GB / 80GB per GPU (64% usage, ~28GB free)
#
# Batch size calculation:
# - Per-device batch: 2 (can increase with 28GB free memory)
# - Gradient accumulation: 8
# - Total effective batch: 2 * 8 * NUM_GPUS sequences
# - Tokens per step: effective_batch * 16384 tokens
PER_DEVICE_BATCH_SIZE=2
GRAD_ACCUM=8

# 정확한 토큰 계산 (16K context)
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
TOKENS_PER_STEP=$((EFFECTIVE_BATCH * 16384))

echo "Batch configuration (16K context, 0.5B model):"
echo "  Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Total GPUs: $NUM_GPUS"
echo "  Effective batch size: $EFFECTIVE_BATCH sequences"
echo "  Tokens per step: $TOKENS_PER_STEP (~$((TOKENS_PER_STEP / 1000))K)"
echo ""

# =============================================================================
# 학습 스텝 설정
# =============================================================================
# Chinchilla optimal for 0.5B model: 10B tokens
# Required steps for 10B: 10,000,000,000 / TOKENS_PER_STEP
#
# 2 GPU 기준:
#   Tokens per step = 2 * 16 * 16384 = 524,288
#   Steps for 10B = 10,000,000,000 / 524,288 ≈ 19,074 steps

# 10B 토큰을 위한 정확한 스텝 계산
MAX_STEPS=$(((10000000000 + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))

# 정확한 총 토큰 수 계산
TOTAL_TOKENS=$((MAX_STEPS * TOKENS_PER_STEP))
TOTAL_TOKENS_B=$(awk -v t="$TOTAL_TOKENS" 'BEGIN {printf "%.2f", t/1e9}')
echo "Training configuration:"
echo "  Target: 10B tokens (Chinchilla optimal for 0.5B model)"
echo "  Tokens per step: $TOKENS_PER_STEP"
echo "  Required steps: $MAX_STEPS"
echo "  Total tokens: $TOTAL_TOKENS (~${TOTAL_TOKENS_B}B)"
echo "  Context length: 16384 tokens"
echo "  Gradient checkpointing: enabled"
echo ""

# =============================================================================
# 체크포인트 확인 (자동 재개)
# =============================================================================
OUTPUT_DIR="./kogum-0.5B-16k-pretrain"
RESUME_ARG=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Found checkpoint: $LATEST_CHECKPOINT"
        echo "Will resume training from this checkpoint"
        RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
    else
        echo "No checkpoints found. Starting from scratch."
    fi
else
    echo "No output directory found. Starting from scratch."
fi
echo ""

# =============================================================================
# 분산 학습 실행
# =============================================================================
echo "Starting distributed training with torchrun..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -m kogum.train.pretrain \
    --config_path src/kogum/configs/kogum_0.5B_16k_kormo.yaml \
    --tokenizer_name KORMo-Team/KORMo-tokenizer \
    --output_dir $OUTPUT_DIR \
    --run_name kogum-0.5B-16k-pretrain \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --save_steps 500 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    $RESUME_ARG

echo ""
echo "========================================"
echo "Distributed pre-training completed!"
echo "========================================"
