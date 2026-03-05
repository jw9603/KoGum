#!/usr/bin/env python3
"""특정 step의 배치를 재현하는 스크립트

비스트리밍 모드에서는 seed와 step 번호만 있으면 정확히 동일한 배치를 재현할 수 있습니다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer, set_seed

from kogum.data_utils import DataCollatorForPackedSequences, tokenize_and_pack


def reproduce_batch_at_step(
    step: int,
    seed: int = 42,
    tokenizer_path: str = "./kogum-tokenizer",
    max_seq_length: int = 4096,
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 32,
    num_gpus: int = 4,
    num_samples: int = 5,
):
    """특정 step의 배치를 재현하고 샘플을 출력

    Args:
        step: 재현할 global step 번호
        seed: Random seed (학습 시와 동일해야 함)
        tokenizer_path: 토크나이저 경로
        max_seq_length: 최대 시퀀스 길이
        per_device_batch_size: 디바이스당 배치 크기
        gradient_accumulation_steps: Gradient accumulation 스텝
        num_gpus: GPU 개수
        num_samples: 출력할 샘플 수
    """

    print("=" * 80)
    print(f"Reproducing batch at step {step}")
    print("=" * 80)

    # Seed 설정
    set_seed(seed)

    # 토크나이저 로드
    print(f"\n[1/4] Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 데이터셋 로드 (비스트리밍)
    print(f"\n[2/4] Loading datasets (non-streaming mode)...")
    korean_dataset = load_dataset(
        "KORMo-Team/korean-web-collection",
        split="train",
        streaming=False,  # 비스트리밍 모드
    )

    english_dataset = load_dataset(
        "KORMo-Team/dclm-baseline-filtered",
        split="train",
        streaming=False,  # 비스트리밍 모드
    )

    print(f"  Korean dataset: {len(korean_dataset)} samples")
    print(f"  English dataset: {len(english_dataset)} samples")

    # Interleave
    dataset = interleave_datasets(
        [korean_dataset, english_dataset],
        seed=seed,
        stopping_strategy="all_exhausted",
    )

    print(f"  Mixed dataset ready")

    # 학습 데이터로 분할 (eval 1000개 제외)
    split_dataset = dataset.train_test_split(test_size=1000, seed=seed)
    train_dataset = split_dataset["train"]

    print(f"  Train dataset: {len(train_dataset)} samples")

    # 토크나이징 및 패킹
    print(f"\n[3/4] Tokenizing and packing...")
    packed_dataset = tokenize_and_pack(
        train_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_proc=8,
    )

    print(f"  Packed dataset: {len(packed_dataset)} sequences")

    # 해당 step의 배치 인덱스 계산
    # step = (batch_idx * gradient_accumulation_steps * num_gpus) / per_device_batch_size
    # 따라서 batch_idx = step * per_device_batch_size / (gradient_accumulation_steps * num_gpus)

    samples_per_step = per_device_batch_size * gradient_accumulation_steps * num_gpus
    start_idx = step * samples_per_step

    print(f"\n[4/4] Extracting batch at step {step}...")
    print(f"  Samples per step: {samples_per_step}")
    print(f"  Start index: {start_idx}")

    # 배치 추출
    if start_idx >= len(packed_dataset):
        print(f"  ⚠️  Error: start_idx {start_idx} >= dataset size {len(packed_dataset)}")
        return

    batch_data = []
    for i in range(min(num_samples, samples_per_step)):
        idx = start_idx + i
        if idx >= len(packed_dataset):
            break

        sample = packed_dataset[idx]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # 디코드
        text = tokenizer.decode(input_ids, skip_special_tokens=False)

        batch_data.append({
            "idx": idx,
            "input_ids_length": len(input_ids),
            "text": text,
            "text_preview": text[:500],
        })

    # 출력
    print("\n" + "=" * 80)
    print(f"Batch samples at step {step}:")
    print("=" * 80)

    for i, sample in enumerate(batch_data):
        print(f"\n[Sample {i + 1}/{len(batch_data)}]")
        print(f"  Dataset index: {sample['idx']}")
        print(f"  Input length: {sample['input_ids_length']} tokens")
        print(f"  Text preview:")
        print(f"    {sample['text_preview']}")
        print("-" * 80)

    # 통계
    avg_length = sum(s["input_ids_length"] for s in batch_data) / len(batch_data)
    print(f"\nBatch statistics:")
    print(f"  Total samples shown: {len(batch_data)}")
    print(f"  Average length: {avg_length:.1f} tokens")

    return batch_data


def main():
    parser = argparse.ArgumentParser(description="Reproduce batch at specific step")
    parser.add_argument("--step", type=int, required=True, help="Global step number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_path", type=str, default="./kogum-tokenizer", help="Tokenizer path")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--save", type=str, default=None, help="Save samples to JSON file")

    args = parser.parse_args()

    batch_data = reproduce_batch_at_step(
        step=args.step,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        max_seq_length=args.max_seq_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_gpus=args.num_gpus,
        num_samples=args.num_samples,
    )

    # JSON 저장
    if args.save and batch_data:
        import json
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to {args.save}")


if __name__ == "__main__":
    main()
