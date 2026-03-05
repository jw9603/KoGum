"""Training arguments for KoGum pre-training and fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class KoGumTrainingArguments(TrainingArguments):
    """KoGum 학습 인자.

    HuggingFace TrainingArguments를 확장하여 KoGum 특화 설정 추가.
    """

    # =========================================================================
    # 기본 학습 설정
    # =========================================================================
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "학습 실행 여부"},
    )
    do_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "평가 실행 여부 (Pre-training에서는 보통 False)"},
    )

    # =========================================================================
    # 정밀도 및 성능
    # =========================================================================
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "BFloat16 mixed precision 사용 (A100/H100 권장)"},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Float16 mixed precision (V100 등 구형 GPU용)"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient checkpointing (메모리 절약, 속도 감소)"},
    )

    # =========================================================================
    # 학습 스케줄
    # =========================================================================
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "학습 에폭 수"},
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "최대 학습 스텝 (-1이면 num_train_epochs 사용)"},
    )

    # =========================================================================
    # 배치 크기 및 그래디언트 누적
    # =========================================================================
    per_device_train_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "GPU당 배치 크기"},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "평가 시 GPU당 배치 크기"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "그래디언트 누적 스텝 (effective batch = batch_size × accum_steps × n_gpus)"},
    )

    # =========================================================================
    # 학습률 및 스케줄러
    # =========================================================================
    learning_rate: Optional[float] = field(
        default=5e-4,
        metadata={"help": "초기 학습률"},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "LR 스케줄러: linear, cosine, constant, warmup_stable_decay 등"},
    )
    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": "Warmup 비율 (전체 스텝의 3%)"},
    )
    warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Warmup 스텝 수 (warmup_ratio 우선)"},
    )

    # =========================================================================
    # Optimizer
    # =========================================================================
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "Optimizer: adamw_torch, adamw_8bit, adafactor 등"},
    )
    weight_decay: Optional[float] = field(
        default=0.1,
        metadata={"help": "Weight decay (L2 정규화)"},
    )
    adam_beta1: Optional[float] = field(
        default=0.9,
        metadata={"help": "Adam beta1 (momentum)"},
    )
    adam_beta2: Optional[float] = field(
        default=0.95,
        metadata={"help": "Adam beta2 (variance) - 0.999보다 낮게 설정 (LLaMA 스타일)"},
    )
    adam_epsilon: Optional[float] = field(
        default=1e-8,
        metadata={"help": "Adam epsilon"},
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={"help": "Gradient clipping (발산 방지)"},
    )

    # =========================================================================
    # 로깅
    # =========================================================================
    logging_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "로깅 전략: steps, epoch"},
    )
    logging_steps: Optional[int] = field(
        default=10,
        metadata={"help": "로깅 주기 (steps)"},
    )
    logging_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": "첫 스텝 로깅 여부"},
    )
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={"help": "로깅 플랫폼: tensorboard, wandb, none"},
    )

    # =========================================================================
    # 체크포인트
    # =========================================================================
    save_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "저장 전략: steps, epoch, no"},
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "체크포인트 저장 주기 (steps)"},
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "최대 체크포인트 개수 (디스크 공간 절약)"},
    )
    save_only_model: Optional[bool] = field(
        default=False,
        metadata={"help": "모델만 저장 (optimizer state 제외). resume 시 LR scheduler 복원을 위해 False 권장."},
    )

    # =========================================================================
    # 평가
    # =========================================================================
    eval_strategy: Optional[str] = field(
        default="no",
        metadata={"help": "평가 전략: steps, epoch, no"},
    )
    eval_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "평가 주기 (steps)"},
    )

    # =========================================================================
    # 분산 학습
    # =========================================================================
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={"help": "DDP에서 사용되지 않는 파라미터 찾기 (보통 False)"},
    )
    fsdp: Optional[str] = field(
        default="",
        metadata={"help": "FSDP 설정 (대용량 모델용)"},
    )
    fsdp_config: Optional[dict] = field(
        default=None,
        metadata={"help": "FSDP 상세 설정"},
    )

    # =========================================================================
    # 기타
    # =========================================================================
    overwrite_output_dir: Optional[bool] = field(
        default=True,
        metadata={"help": "출력 디렉토리 덮어쓰기 허용"},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "사용하지 않는 컬럼 자동 제거 (causal LM에서는 False 권장)"},
    )
    include_num_input_tokens_seen: Optional[bool] = field(
        default=True,
        metadata={"help": "입력 토큰 수 추적 (학습 진행도 확인용)"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "체크포인트 재개 경로"},
    )

    # =========================================================================
    # KoGum 특화 설정
    # =========================================================================
    run_name: Optional[str] = field(
        default="kogum-pretrain",
        metadata={"help": "실험 이름 (로깅/체크포인트 구분용)"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "랜덤 시드 (재현성)"},
    )
