#!/bin/bash
# KoGum 학습 환경 설정 스크립트
# 새로운 서버에서 실행하여 환경을 구성합니다.

set -e

echo "========================================"
echo "KoGum Environment Setup"
echo "========================================"

# =============================================================================
# 1. Conda 환경 확인/생성
# =============================================================================
# echo ""
# echo "[1/5] Checking Conda environment..."

# if conda env list | grep -q "^kogum "; then
#     echo "✅ Conda environment 'kogum' already exists"
# else
#     echo "Creating Conda environment 'kogum' (Python 3.10)..."
#     conda create -n kogum python=3.10 -y
#     echo "✅ Conda environment created"
# fi

# =============================================================================
# 2. 패키지 설치
# =============================================================================
echo ""
echo "[2/5] Installing dependencies..."

# Conda 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate kogum

# KoGum 패키지 설치 (editable mode)
pip install -e .

echo "✅ Dependencies installed"

# =============================================================================
# 3. GPU 확인
# =============================================================================
echo ""
echo "[3/5] Checking GPU availability..."

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "✅ NVIDIA GPUs detected: $NUM_GPUS"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi

# PyTorch GPU 확인
python -c "
import torch
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(f'\nPyTorch CUDA available: {cuda_available}')
print(f'PyTorch visible GPUs: {num_gpus}')
if cuda_available:
    for i in range(num_gpus):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# =============================================================================
# 4. 환경 변수 설정
# =============================================================================
echo ""
echo "[4/5] Setting up environment variables..."

if [ -f .env ]; then
    echo "✅ .env file already exists"
else
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys:"
    echo "   - WANDB_API_KEY"
    echo "   - HF_TOKEN (optional)"
fi

# =============================================================================
# 5. 토크나이저 확인
# =============================================================================
echo ""
echo "[5/5] Checking tokenizer..."

if [ -d "kogum-tokenizer" ]; then
    echo "✅ Tokenizer found at ./kogum-tokenizer"
    python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./kogum-tokenizer')
print(f'  Vocab size: {tokenizer.vocab_size:,}')
print(f'  Special tokens: {list(tokenizer.special_tokens_map.keys())}')
"
else
    echo "⚠️  Warning: Tokenizer not found at ./kogum-tokenizer"
    echo "   You may need to:"
    echo "   1. Copy from another server: scp -r user@server:/path/to/kogum-tokenizer ."
    echo "   2. Or train a new tokenizer: python scripts/train_tokenizer.py [args]"
fi

# =============================================================================
# 완료
# =============================================================================
echo ""
echo "========================================"
echo "✅ Environment setup completed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate kormo"
echo "  2. Edit .env file: nano .env"
echo "  3. Run training: bash scripts/run_pretrain_distributed.sh"
echo ""
