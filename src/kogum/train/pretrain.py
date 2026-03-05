# -*- coding: utf-8 -*-
"""KoGum Pre-training Script with KORMo Tokenizer.

KoGum 모델을 KORMo tokenizer로 from scratch 학습하는 스크립트입니다.

Usage:
    torchrun --nproc_per_node=2 -m kogum.train.pretrain \
        --config_path src/kogum/configs/kogum_0.5B_16k_kormo.yaml
"""

import os
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

# NCCL XML 파일 문제 해결 (GPU 2,3 사용 시 필수)
# Azure NCv4의 기본 topo.xml은 GPU 0-3을 모두 참조
# CUDA_VISIBLE_DEVICES=2,3 사용 시 GPU 0,1이 없어서 에러 발생
# GPU 2,3만 포함하는 커스텀 XML 파일로 override (dev 0,1로 리매핑됨)
os.environ['NCCL_TOPO_FILE'] = '/tmp/custom_nccl_topo.xml'
os.environ['NCCL_GRAPH_FILE'] = '/tmp/custom_nccl_graph.xml'

from kogum.data_utils import (
    DataCollatorWithDocumentBoundaries,
    pack_dataset,
    tokenize_and_pack,
)
from kogum.model import KoGumConfig, KoGumForCausalLM
from kogum.train import (
    KoGumTrainer,
    KoGumTrainingArguments,
)
from kogum.train.spike_detector import BatchSpikeDetector


def print_rank0(*args, **kwargs):
    """분산 학습 시 메인 프로세스(rank 0)에서만 출력합니다."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def load_config_from_yaml(yaml_path: str) -> KoGumConfig:
    """YAML 파일에서 모델 config를 로드합니다."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return KoGumConfig(**config_dict)


def load_and_mix_datasets(streaming: bool, seed: int):
    """한국어(70%)와 영어(30%) 데이터셋을 로드하고 interleave합니다.

    처리 순서:
        1. 한국어/영어 데이터셋을 HuggingFace Hub에서 streaming으로 로드
        2. 분산 학습 시 각 GPU(rank)가 서로 다른 샤드를 읽도록 분할
        3. interleave_datasets로 70:30 비율로 혼합

    Args:
        streaming: True면 IterableDataset으로 로드 (디스크에 저장 안 함)
        seed: interleave 시 랜덤 시드 (재현성)

    Returns:
        interleave된 데이터셋 (각 sample = {"text": "..."})
    """
    from datasets import interleave_datasets

    # 분산 학습 시 데이터 샤딩
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    print_rank0(f"  Loading Korean dataset: KORMo-Team/korean-web-collection")
    korean_dataset = load_dataset(
        "KORMo-Team/korean-web-collection",
        split="train",
        streaming=streaming,
    )

    # Streaming 모드에서는 각 rank가 다른 샤드를 읽어야 함
    if streaming and world_size > 1:
        korean_dataset = korean_dataset.shard(num_shards=world_size, index=rank)
        print_rank0(f"  Sharded Korean dataset: rank {rank}/{world_size}")

    print_rank0(f"  Loading English dataset: KORMo-Team/dclm-baseline-filtered")
    english_dataset = load_dataset(
        "KORMo-Team/dclm-baseline-filtered",
        split="train",
        streaming=streaming,
    )

    # Streaming 모드에서는 각 rank가 다른 샤드를 읽어야 함
    if streaming and world_size > 1:
        english_dataset = english_dataset.shard(num_shards=world_size, index=rank)
        print_rank0(f"  Sharded English dataset: rank {rank}/{world_size}")

    # 70% Korean, 30% English
    print_rank0(f"  Mixing datasets: 70% Korean, 30% English")
    combined_dataset = interleave_datasets(
        [korean_dataset, english_dataset],
        probabilities=[0.7, 0.3],
        seed=seed,
        stopping_strategy="all_exhausted" if not streaming else "first_exhausted",
    )

    return combined_dataset


def main():
    """메인 학습 함수."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/kogum/configs/kogum_0.5B_16k_kormo.yaml",
        help="Model config YAML file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kogum-0.5B-16k-pretrain",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="KORMo-Team/KORMo-tokenizer",
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="kogum-0.5B-16k-pretrain",
        help="Run name for wandb"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=4768,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every N steps"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Reporting tool (wandb, tensorboard, etc.)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    # Seed 설정
    set_seed(args.seed)

    print_rank0("=" * 80)
    print_rank0("KoGum 0.5B Pre-training with KORMo Tokenizer")
    print_rank0("=" * 80)

    # 1. Config 로드
    print_rank0("\n[1/6] Loading model config...")
    print_rank0(f"  Config file: {args.config_path}")
    model_config = load_config_from_yaml(args.config_path)
    print_rank0(f"  ✓ Vocab size: {model_config.vocab_size:,}")
    print_rank0(f"  ✓ Hidden size: {model_config.hidden_size}")
    print_rank0(f"  ✓ Layers: {model_config.num_hidden_layers}")

    # 2. Tokenizer 로드
    print_rank0(f"\n[2/6] Loading tokenizer...")
    print_rank0(f"  Tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print_rank0(f"  ✓ Vocab size: {tokenizer.vocab_size:,}")
    print_rank0(f"  ✓ PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
    print_rank0(f"  ✓ BOS: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
    print_rank0(f"  ✓ EOS: {tokenizer.eos_token} ({tokenizer.eos_token_id})")

    # Tokenizer vocab size 검증
    if tokenizer.vocab_size != model_config.vocab_size - 41:  # KORMo has 41 special tokens
        print_rank0(f"  ⚠ Warning: tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({model_config.vocab_size})")

    # 3. Model 초기화
    print_rank0(f"\n[3/6] Initializing model from scratch...")
    model = KoGumForCausalLM(model_config)

    # NOTE: Do NOT manually convert model to bfloat16 here
    # Trainer will handle dtype conversion when bf16=True is set in TrainingArguments
    # Manual conversion can interfere with DDP and mixed precision training

    num_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"  ✓ Total parameters: {num_params:,} ({num_params / 1e9:.2f}B)")
    print_rank0(f"  ✓ Model dtype: {next(model.parameters()).dtype}")

    # 4. Dataset 로드
    print_rank0(f"\n[4/6] Loading and processing dataset...")
    streaming = True
    train_dataset = load_and_mix_datasets(streaming=streaming, seed=args.seed)

    # Tokenization + packing with document boundaries
    print_rank0(f"  Tokenizing and packing sequences with document boundaries...")
    seq_len = model_config.max_position_embeddings

    # Step 1: Tokenize
    def tokenize_fn(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        # Add EOS token at the end of each document
        if tokenizer.eos_token_id is not None:
            tokenized["input_ids"] = [
                ids + [tokenizer.eos_token_id] for ids in tokenized["input_ids"]
            ]
        return tokenized

    from datasets import IterableDataset
    is_iterable = isinstance(train_dataset, IterableDataset)

    if is_iterable:
        print_rank0(f"  Tokenizing (streaming mode)...")
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=train_dataset.column_names,
        )
    else:
        print_rank0(f"  Tokenizing {len(train_dataset):,} samples...")
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=train_dataset.column_names,
            num_proc=8,
            desc="Tokenizing",
        )

    # Step 2: Pack with document boundaries
    print_rank0(f"  Packing sequences (seq_len={seq_len})...")
    train_dataset = pack_dataset(
        train_dataset,
        seq_len=seq_len,
        num_proc=None if is_iterable else 8,
        batch_size=10000,
        with_boundaries=False,  # TEMPORARILY DISABLED: intra-document masking causing CUDA error
        eos_token_id=tokenizer.eos_token_id,
    )

    print_rank0(f"  ✓ Sequence length: {seq_len}")
    print_rank0(f"  ✓ Streaming mode: {streaming}")
    print_rank0(f"  ✓ Intra-document causal masking: disabled (testing)")

    # Data collator with document boundaries
    data_collator = DataCollatorWithDocumentBoundaries(tokenizer=tokenizer)

    # 5. Training Arguments
    print_rank0(f"\n[5/6] Setting up training arguments...")
    training_args = KoGumTrainingArguments(
        # Output
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        run_name=args.run_name,

        # Training
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimization
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,

        # Mixed precision
        bf16=args.bf16,
        tf32=True,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        report_to=[args.report_to] if args.report_to else [],

        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,

        # Distributed
        ddp_find_unused_parameters=False,

        # Gradient Checkpointing (for 32K context)
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Gradient Clipping
        max_grad_norm=args.max_grad_norm,

        # Other
        seed=args.seed,
        dataloader_num_workers=4,
    )

    print_rank0(f"  ✓ Max steps: {training_args.max_steps:,}")
    print_rank0(f"  ✓ Batch size: {training_args.per_device_train_batch_size}")
    print_rank0(f"  ✓ Grad accumulation: {training_args.gradient_accumulation_steps}")
    print_rank0(f"  ✓ Learning rate: {training_args.learning_rate}")
    print_rank0(f"  ✓ Warmup ratio: {training_args.warmup_ratio}")

    # Spike detector 설정
    print_rank0(f"\n  Setting up spike detector...")
    batch_spike_detector = BatchSpikeDetector(
        output_dir=os.path.join(args.output_dir, "spike_batches"),
        loss_threshold=2.0,  # 개선된 threshold
        grad_norm_threshold=2.0,
        window_size=10,
        consecutive_steps=2,  # 2 스텝 연속 필요
        save_batch_samples=5,
    )
    print_rank0(f"  ✓ Loss threshold: 2.0x")
    print_rank0(f"  ✓ Grad norm threshold: 2.0x")
    print_rank0(f"  ✓ Consecutive steps: 2")

    # 6. Trainer 초기화
    print_rank0(f"\n[6/6] Initializing trainer...")
    trainer = KoGumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        spike_detector=batch_spike_detector,
    )

    print_rank0(f"  ✓ Trainer initialized")

    # 학습 시작
    print_rank0("\n" + "=" * 80)
    print_rank0("Starting training...")
    print_rank0("=" * 80)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 최종 모델 저장
    print_rank0("\n" + "=" * 80)
    print_rank0("Saving final model...")
    print_rank0("=" * 80)

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print_rank0(f"  ✓ Model saved to: {final_model_dir}")

    print_rank0("\n" + "=" * 80)
    print_rank0("Training completed!")
    print_rank0("=" * 80)


if __name__ == "__main__":
    main()
