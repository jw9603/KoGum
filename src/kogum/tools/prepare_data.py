"""데이터셋 준비 및 확인 스크립트.

학습에 사용할 데이터셋을 로드하고 기본 통계를 확인합니다.

Usage:
    python scripts/prepare_data.py \
        --dataset_name KORMo-Team/korean-web-collection \
        --num_samples 1000
"""

import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="데이터셋 준비 및 확인")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="KORMo-Team/korean-web-collection",
        help="데이터셋 이름",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="확인할 샘플 수",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="텍스트 컬럼명",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("데이터셋 준비 및 확인")
    print("=" * 70)
    print(f"\nDataset: {args.dataset_name}")
    print(f"Samples to check: {args.num_samples:,}")

    # 데이터셋 로드 (streaming mode)
    print("\n[1/3] Loading dataset (streaming mode)...")
    try:
        ds = load_dataset(args.dataset_name, split="train", streaming=True)
        print("✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # 샘플 확인
    print(f"\n[2/3] Checking first {args.num_samples} samples...")
    text_lengths = []
    valid_samples = 0

    for i, example in enumerate(ds.take(args.num_samples)):
        if i >= args.num_samples:
            break

        text = example.get(args.text_column, "")
        if text and len(text.strip()) > 0:
            text_lengths.append(len(text))
            valid_samples += 1

            # 첫 3개 샘플 출력
            if i < 3:
                print(f"\n[Sample {i+1}]")
                print(f"  Length: {len(text)} chars")
                print(f"  Preview: {text[:200]}...")

    # 통계
    print(f"\n[3/3] Statistics")
    print(f"  Valid samples: {valid_samples:,} / {args.num_samples:,}")
    if text_lengths:
        print(f"  Avg length: {sum(text_lengths) / len(text_lengths):.1f} chars")
        print(f"  Min length: {min(text_lengths)} chars")
        print(f"  Max length: {max(text_lengths)} chars")

    print("\n" + "=" * 70)
    print("✅ Dataset is ready for training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
