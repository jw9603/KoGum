#!/bin/bash
# KoGum 0.5B (16K context) Mid-training Script
# 목적: 사전학습된 checkpoint-12500 기반 continued pre-training
# 데이터: 한국어 70% + 영어 30% (수학/코드 비중 강화)
# 목표: 1B tokens

set -e

source ~/.bashrc
conda activate kogum

PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# .env 파일 로드
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# PyTorch CUDA 메모리 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# NCCL 설정
export NCCL_TOPO_FILE=/tmp/custom_nccl_topo.xml
export NCCL_GRAPH_FILE=/tmp/custom_nccl_graph.xml
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

echo "========================================"
echo "KoGum 0.5B (16K) Mid-training"
echo "========================================"

# =============================================================================
# 하드웨어 설정
# =============================================================================
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "❌ CUDA_VISIBLE_DEVICES is not set. Please set it in .env file."
    exit 1
fi

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Using GPUs: $NUM_GPUS"

if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "❌ Invalid CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    exit 1
fi

# =============================================================================
# 배치 크기 설정 (사전학습과 동일)
# =============================================================================
PER_DEVICE_BATCH_SIZE=2
GRAD_ACCUM=8

EFFECTIVE_BATCH=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))
TOKENS_PER_STEP=$((EFFECTIVE_BATCH * 16384))

echo "Batch configuration:"
echo "  Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Total GPUs: $NUM_GPUS"
echo "  Effective batch size: $EFFECTIVE_BATCH sequences"
echo "  Tokens per step: $TOKENS_PER_STEP (~$((TOKENS_PER_STEP / 1000))K)"
echo ""

# =============================================================================
# 학습 스텝 설정 (1B tokens)
# =============================================================================
# 2 GPU 기준:
#   Tokens per step = 2 * 8 * 2 * 16384 = 524,288
#   Steps for 1B = 1,000,000,000 / 524,288 ≈ 1,907 steps
MAX_STEPS=$(((1000000000 + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))

TOTAL_TOKENS=$((MAX_STEPS * TOKENS_PER_STEP))
TOTAL_TOKENS_B=$(awk -v t="$TOTAL_TOKENS" 'BEGIN {printf "%.2f", t/1e9}')

echo "Mid-training configuration:"
echo "  Base model: checkpoint-12500"
echo "  Target: 1B tokens"
echo "  Learning rate: 5e-5 (1/6 of pre-training 3e-4)"
echo "  Tokens per step: $TOKENS_PER_STEP"
echo "  Required steps: $MAX_STEPS"
echo "  Total tokens: $TOTAL_TOKENS (~${TOTAL_TOKENS_B}B)"
echo ""

# =============================================================================
# 모델 경로 및 체크포인트 확인
# =============================================================================
MODEL_PATH="./checkpoints_backup/checkpoint-12500"
OUTPUT_DIR="./kogum-0.5B-16k-midtrain"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model checkpoint not found: $MODEL_PATH"
    exit 1
fi
echo "Base model: $MODEL_PATH"

RESUME_ARG=""
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Found checkpoint: $LATEST_CHECKPOINT"
        echo "Will resume training from this checkpoint"
        RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
    fi
fi
echo ""

# =============================================================================
# 분산 학습 실행
# =============================================================================
echo "Starting mid-training with torchrun..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    -m kogum.train.midtrain \
    --model_name_or_path $MODEL_PATH \
    --config_path src/kogum/configs/kogum_0.5B_16k_kormo.yaml \
    --tokenizer_name KORMo-Team/KORMo-tokenizer \
    --output_dir $OUTPUT_DIR \
    --run_name kogum-0.5B-16k-midtrain \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --save_steps 100 \
    --save_total_limit 10 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    $RESUME_ARG

echo ""
echo "========================================"
echo "Mid-training completed!"
echo "========================================"