#!/bin/bash
# KoGum 0.5B (16K context) SFT Script
# 목적: Mid-training된 모델을 instruction-following 데이터로 fine-tuning
# 데이터: 한국어 2개 + 영어 2개 (non-streaming)
# - KORMo-Team/NemoPost-ko-synth-sft
# - KORMo-Team/IF-bilingual-sft
# - nvidia/Nemotron-Post-Training-Dataset-v1
# - HuggingFaceTB/smoltalk2

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
export CUDA_LAUNCH_BLOCKING=1
# transformers 4.57+ torch.load 보안 체크 우회 (PyTorch 2.5.1 사용 중)
export TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION=1

# libnvrtc 경로 설정
export LD_LIBRARY_PATH=/mdisk/jiwon_jeong/anaconda3/envs/kogum/lib:$LD_LIBRARY_PATH

# NCCL 설정
# NCCL_TOPO/GRAPH_FILE은 파일이 실제로 존재할 때만 설정
if [ -f /tmp/custom_nccl_topo.xml ]; then
    export NCCL_TOPO_FILE=/tmp/custom_nccl_topo.xml
fi
if [ -f /tmp/custom_nccl_graph.xml ]; then
    export NCCL_GRAPH_FILE=/tmp/custom_nccl_graph.xml
fi
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

echo "========================================"
echo "KoGum 0.5B (16K) SFT"
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
# 배치 크기 설정
# =============================================================================
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM=16

EFFECTIVE_BATCH=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

echo "Batch configuration:"
echo "  Per-device batch size: $PER_DEVICE_BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Total GPUs: $NUM_GPUS"
echo "  Effective batch size: $EFFECTIVE_BATCH sequences"
echo ""

# =============================================================================
# 모델 경로 및 체크포인트 확인
# =============================================================================
MODEL_PATH="./kogum-0.5B-16k-sft/checkpoint-8000"
OUTPUT_DIR="./kogum-0.5B-16k-sft-ep2"

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
# SFT 설정 출력
# =============================================================================
echo "SFT configuration:"
echo "  Base model: $MODEL_PATH"
echo "  Epochs: 1"
echo "  Learning rate: 2e-6"
echo "  Max length: 16384"
echo "  Gradient checkpointing: enabled"
echo ""

# =============================================================================
# 분산 학습 실행
# =============================================================================
echo "Starting SFT with torchrun..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    -m kogum.train.sft \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name KORMo-Team/KORMo-tokenizer \
    --output_dir $OUTPUT_DIR \
    --run_name kogum-0.5B-16k-sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 2e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --lr_scheduler_type cosine \
    --max_length 16384 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 10 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    $RESUME_ARG \
    2>&1 | tee /tmp/sft_train.log

echo ""
echo "========================================"
echo "SFT completed!"
echo "========================================"