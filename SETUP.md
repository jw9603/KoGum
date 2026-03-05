# KoGum 새 서버 설정 가이드

새로운 서버(4 GPU)에서 KoGum 학습 환경을 설정하는 방법입니다.

## 빠른 시작 (자동 설정)

```bash
# 1. 저장소를 새 서버로 복사
scp -r /path/to/KoGum user@new-server:/path/to/

# 2. 새 서버에 접속
ssh user@new-server

# 3. 프로젝트 디렉토리로 이동
cd /path/to/KoGum

# 4. 자동 설정 스크립트 실행
bash scripts/setup_environment.sh
```

설정 스크립트가 자동으로:
- ✅ Conda 환경 생성 (`kormo`)
- ✅ 의존성 패키지 설치
- ✅ GPU 확인
- ✅ `.env` 파일 생성
- ✅ 토크나이저 확인

## 수동 설정 (단계별)

자동 스크립트가 작동하지 않는 경우:

### 1. Conda 환경 생성

```bash
conda create -n kormo python=3.10 -y
conda activate kormo
```

### 2. 의존성 설치

```bash
# pyproject.toml 기반 설치 (editable mode)
pip install -e .
```

**설치되는 주요 패키지**:
- `torch>=2.0.0` (CUDA 지원)
- `transformers>=4.40.0`
- `datasets>=2.18.0`
- `tokenizers>=0.15.0`
- `wandb>=0.16.0`
- 기타 (tqdm, pyyaml, etc.)

### 3. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
```

**필수 설정**:
```bash
# WandB (학습 모니터링)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=kogum-pretrain
WANDB_ENTITY=jwjw9603-no

# HuggingFace (선택)
HF_TOKEN=your_hf_token_here

# GPU 설정 (필요시)
CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPU
```

**WandB API 키 확인**: https://wandb.ai/authorize

### 4. 토크나이저 복사

토크나이저는 이미 학습되어 있으므로 복사만 하면 됩니다:

```bash
# 기존 서버에서
scp -r ./kogum-tokenizer user@new-server:/path/to/KoGum/

# 또는 tar.gz로 압축하여 전송
tar -czf kogum-tokenizer.tar.gz kogum-tokenizer/
scp kogum-tokenizer.tar.gz user@new-server:/path/to/KoGum/
# 새 서버에서 압축 해제
tar -xzf kogum-tokenizer.tar.gz
```

### 5. GPU 확인

```bash
# GPU 정보 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

**예상 출력** (4 GPU):
```
CUDA: True, GPUs: 4
```

## 학습 시작

환경 설정이 완료되면:

```bash
# Conda 환경 활성화
conda activate kormo

# 분산 학습 시작 (4 GPU)
bash scripts/run_pretrain_distributed.sh
```

**또는** tmux로 백그라운드 실행:

```bash
# tmux 세션 생성
tmux new -s kogum-train

# 학습 실행
bash scripts/run_pretrain_distributed.sh

# tmux 세션 detach: Ctrl+B, D
# tmux 세션 재접속: tmux attach -t kogum-train
```

## 트러블슈팅

### `pip install -e .` 실패

```bash
# pip 업그레이드
pip install --upgrade pip setuptools wheel

# 다시 시도
pip install -e .
```

### CUDA 버전 불일치

```bash
# 현재 CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch 재설치 (CUDA 11.8 예시)
pip uninstall torch -y
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 토크나이저 없음

```bash
# 기존 서버에서 복사하거나
scp -r user@old-server:/path/to/kogum-tokenizer .

# 새로 학습 (권장하지 않음, 시간 소요)
python scripts/train_tokenizer.py [args]
```

### WandB 로그인 실패

```bash
# WandB 재로그인
wandb login --relogin

# API 키 입력
```

### OOM (Out of Memory)

`run_pretrain_distributed.sh`에서 배치 크기 조정:

```bash
# 기본값: PER_DEVICE_BATCH_SIZE=8
# 메모리 부족 시 줄이기:
PER_DEVICE_BATCH_SIZE=6  # 또는 4

# Gradient accumulation은 자동으로 조정됨
```

## 파일 체크리스트

새 서버로 복사해야 할 파일/디렉토리:

- ✅ `src/` - 소스 코드
- ✅ `scripts/` - 학습 스크립트
- ✅ `pyproject.toml` - 의존성 정의
- ✅ `kogum-tokenizer/` - 토크나이저 (필수!)
- ✅ `.env.example` - 환경 변수 템플릿
- ⚠️ `.env` - API 키 (보안 주의! 수동으로 재생성 권장)

**복사하지 않아도 되는 것**:
- ❌ `test-run/` - 테스트 체크포인트 (삭제됨)
- ❌ `wandb/` - 로컬 로그
- ❌ `__pycache__/`, `.pyc` - Python 캐시

## 전체 프로세스 요약

```bash
# === 기존 서버 ===
cd KoGum
tar -czf kogum-package.tar.gz \
    src/ scripts/ pyproject.toml \
    kogum-tokenizer/ .env.example README.md TRAINING.md SETUP.md

# 새 서버로 전송
scp kogum-package.tar.gz user@new-server:~/

# === 새 서버 ===
ssh user@new-server
tar -xzf kogum-package.tar.gz
cd KoGum

# 환경 설정
bash scripts/setup_environment.sh

# .env 편집
nano .env  # API 키 입력

# 학습 시작
conda activate kormo
bash scripts/run_pretrain_distributed.sh
```

## 학습 모니터링

- **WandB 대시보드**: https://wandb.ai/jwjw9603-no/kogum-pretrain
- **체크포인트 저장**: `./kogum-pretrain/checkpoint-*/`
- **로그 파일**: `wandb/run-*/logs/`

## 도움말

더 자세한 정보:
- **학습 가이드**: [TRAINING.md](TRAINING.md)
- **모델 구조**: [src/kogum/model/](src/kogum/model/)
- **데이터 처리**: [src/kogum/data_utils/](src/kogum/data_utils/)
