# -*- coding: utf-8 -*-
"""KoGum Mid-training Script.

사전학습된 KoGum 모델을 고품질 데이터로 continued pre-training합니다.
- 영어:한국어 = 65:35 비율
- 수학/코드 추론 데이터 비중 강화
- 11개 데이터셋 개별 비율 interleave

Usage:
    torchrun --nproc_per_node=2 -m kogum.train.midtrain \
        --model_name_or_path ./checkpoints_backup/checkpoint-12500
"""

import os
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

# NCCL XML 파일 문제 해결 (GPU 2,3 사용 시 필수)
os.environ['NCCL_TOPO_FILE'] = '/tmp/custom_nccl_topo.xml'
os.environ['NCCL_GRAPH_FILE'] = '/tmp/custom_nccl_graph.xml'

from kogum.data_utils import (
    DataCollatorWithDocumentBoundaries,
    pack_dataset,
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


def load_midtrain_datasets(streaming: bool, seed: int):
    """Mid-training용 11개 데이터셋을 개별 비율로 interleave합니다.

    영어 (합 0.65):
        UltraFineWeb-filtered: 0.17 (웹)
        smollm-corpus: 0.11 (웹+코드+교과서)
        cosmopedia: 0.07 (합성 교과서)
        finemath: 0.13 (수학)
        OpenMathReasoning: 0.10 (수학 추론)
        OpenCodeReasoning: 0.07 (코드 추론)

    한국어 (합 0.35):
        UltraFineWeb-ko-synth: 0.11 (웹)
        FineWeb2-ko-synth: 0.08 (웹)
        NemoPost-ko-synth: 0.06 (추론)
        Cosmopedia-ko-synth: 0.05 (교과서)
        korean-public-corpus: 0.05 (공공)
    """
    from datasets import interleave_datasets

    # 분산 학습 시 데이터 샤딩
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    def load_and_shard(name, split="train", config=None):
        """데이터셋 로드 후 분산 학습 시 샤딩."""
        print_rank0(f"  Loading: {name}" + (f" ({config})" if config else ""))
        if config:
            ds = load_dataset(name, config, split=split, streaming=streaming)
        else:
            ds = load_dataset(name, split=split, streaming=streaming)
        if streaming and world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        return ds

    # =====================================================================
    # 영어 데이터셋 (합 0.65)
    # =====================================================================
    # 1. UltraFineWeb-filtered (웹, 0.17) - text 컬럼만 사용
    en_ultrafine = load_and_shard("KORMo-Team/UltraFineWeb-filtered")
    en_ultrafine = en_ultrafine.select_columns(["text"])

    # 2. smollm-corpus (웹+코드+교과서, 0.11) - cosmopedia-v2 config
    en_smollm = load_and_shard("HuggingFaceTB/smollm-corpus", config="cosmopedia-v2")
    en_smollm = en_smollm.select_columns(["text"])

    # 3. cosmopedia (합성 교과서, 0.07) - web_samples_v1 config
    en_cosmo = load_and_shard("HuggingFaceTB/cosmopedia", config="web_samples_v1")
    en_cosmo = en_cosmo.select_columns(["text"])

    # 4. finemath (수학, 0.13) - finemath-3plus config
    en_math = load_and_shard("HuggingFaceTB/finemath", config="finemath-3plus")
    en_math = en_math.select_columns(["text"])

    # 5. OpenMathReasoning (수학 추론, 0.10) - cot split
    #    problem + generated_solution → text로 변환
    en_math_reason = load_and_shard("nvidia/OpenMathReasoning", split="cot")
    en_math_reason = en_math_reason.map(
        lambda x: {"text": (x.get("problem") or "") + "\n\n" + (x.get("generated_solution") or "")},
    )
    en_math_reason = en_math_reason.select_columns(["text"])

    # 6. OpenCodeReasoning (코드 추론, 0.07)
    #    input + output → text로 변환
    en_code_reason = load_and_shard("nvidia/OpenCodeReasoning", config="split_0", split="split_0")
    en_code_reason = en_code_reason.map(
        lambda x: {"text": (x.get("input") or "") + "\n\n" + (x.get("output") or "")},
    )
    en_code_reason = en_code_reason.select_columns(["text"])

    # =====================================================================
    # 한국어 데이터셋 (합 0.35)
    # =====================================================================
    # 7. UltraFineWeb-ko-synth (웹, 0.11)
    ko_ultrafine = load_and_shard("KORMo-Team/UltraFineWeb-ko-synth")
    ko_ultrafine = ko_ultrafine.select_columns(["text"])

    # 8. FineWeb2-ko-synth (웹, 0.08)
    ko_fineweb = load_and_shard("KORMo-Team/FineWeb2-ko-synth")
    ko_fineweb = ko_fineweb.select_columns(["text"])

    # 9. NemoPost-ko-synth (추론, 0.06)
    ko_nemo = load_and_shard("KORMo-Team/NemoPost-ko-synth")
    ko_nemo = ko_nemo.select_columns(["text"])

    # 10. Cosmopedia-ko-synth (교과서, 0.05)
    ko_cosmo = load_and_shard("KORMo-Team/Cosmopedia-ko-synth")
    ko_cosmo = ko_cosmo.select_columns(["text"])

    # 11. korean-public-corpus (공공, 0.05)
    ko_public = load_and_shard("KORMo-Team/korean-public-corpus")
    ko_public = ko_public.select_columns(["text"])

    # =====================================================================
    # Interleave (개별 비율)
    # =====================================================================
    datasets = [
        en_ultrafine, en_smollm, en_cosmo, en_math, en_math_reason, en_code_reason,
        ko_ultrafine, ko_fineweb, ko_nemo, ko_cosmo, ko_public,
    ]
    probabilities = [
        0.17, 0.11, 0.07, 0.13, 0.10, 0.07,  # 영어: 합 0.65
        0.11, 0.08, 0.06, 0.05, 0.05,          # 한국어: 합 0.35
    ]

    dataset_names = [
        "UltraFineWeb-filtered", "smollm-corpus", "cosmopedia",
        "finemath", "OpenMathReasoning", "OpenCodeReasoning",
        "UltraFineWeb-ko-synth", "FineWeb2-ko-synth", "NemoPost-ko-synth",
        "Cosmopedia-ko-synth", "korean-public-corpus",
    ]

    print_rank0(f"\n  Dataset mixing ratios:")
    for name, prob in zip(dataset_names, probabilities):
        print_rank0(f"    {name}: {prob:.2f}")

    combined = interleave_datasets(
        datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy="first_exhausted",
    )

    return combined


def main():
    """Mid-training 메인 함수."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Pre-trained model checkpoint path"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/kogum/configs/kogum_0.5B_16k_kormo.yaml",
        help="Model config YAML file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kogum-0.5B-16k-midtrain",
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
        default="kogum-0.5B-16k-midtrain",
        help="Run name for wandb"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1907,
        help="Maximum training steps (1B tokens / 524288 tokens_per_step)"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Peak learning rate (lower than pre-training)"
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
        default=200,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20,
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
        help="Reporting tool"
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
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep"
    )
    args = parser.parse_args()

    # Seed 설정
    set_seed(args.seed)

    print_rank0("=" * 80)
    print_rank0("KoGum 0.5B Mid-training")
    print_rank0("=" * 80)

    # 1. Tokenizer 로드
    print_rank0(f"\n[1/6] Loading tokenizer...")
    print_rank0(f"  Tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print_rank0(f"  ✓ Vocab size: {tokenizer.vocab_size:,}")

    # 2. Model 로드 (사전학습된 체크포인트에서)
    print_rank0(f"\n[2/6] Loading pre-trained model...")
    print_rank0(f"  Checkpoint: {args.model_name_or_path}")
    model = KoGumForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"  ✓ Total parameters: {num_params:,} ({num_params / 1e9:.2f}B)")
    print_rank0(f"  ✓ Model dtype: {next(model.parameters()).dtype}")

    # 3. Dataset 로드
    print_rank0(f"\n[3/6] Loading mid-training datasets...")
    streaming = True
    train_dataset = load_midtrain_datasets(streaming=streaming, seed=args.seed)

    # 4. Tokenization + Packing
    print_rank0(f"\n[4/6] Tokenizing and packing...")
    seq_len = model.config.max_position_embeddings

    def tokenize_fn(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
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
            remove_columns=["text"],
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

    print_rank0(f"  Packing sequences (seq_len={seq_len})...")
    train_dataset = pack_dataset(
        train_dataset,
        seq_len=seq_len,
        num_proc=None if is_iterable else 8,
        batch_size=10000,
        with_boundaries=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    print_rank0(f"  ✓ Sequence length: {seq_len}")
    print_rank0(f"  ✓ Streaming mode: {streaming}")

    # Data collator
    data_collator = DataCollatorWithDocumentBoundaries(tokenizer=tokenizer)

    # 5. Training Arguments
    print_rank0(f"\n[5/6] Setting up training arguments...")
    training_args = KoGumTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        run_name=args.run_name,

        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,

        bf16=args.bf16,
        tf32=True,

        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        report_to=[args.report_to] if args.report_to else [],

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        ddp_find_unused_parameters=False,

        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=args.max_grad_norm,

        seed=args.seed,
        dataloader_num_workers=4,
    )

    print_rank0(f"  ✓ Max steps: {training_args.max_steps:,}")
    print_rank0(f"  ✓ Learning rate: {training_args.learning_rate}")
    print_rank0(f"  ✓ Warmup ratio: {training_args.warmup_ratio}")
    print_rank0(f"  ✓ Save total limit: {args.save_total_limit}")

    # Spike detector
    batch_spike_detector = BatchSpikeDetector(
        output_dir=os.path.join(args.output_dir, "spike_batches"),
        loss_threshold=2.0,
        grad_norm_threshold=2.0,
        window_size=10,
        consecutive_steps=2,
        save_batch_samples=5,
    )

    # 6. Trainer
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
    print_rank0("Starting mid-training...")
    print_rank0(f"  Base model: {args.model_name_or_path}")
    print_rank0(f"  Target: 1B tokens")
    print_rank0(f"  Learning rate: {args.learning_rate} (1/6 of pre-training)")
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
    print_rank0("\nMid-training completed!")


if __name__ == "__main__":
    main()