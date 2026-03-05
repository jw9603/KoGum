#!/usr/bin/env python3
"""학습 중인 데이터 배치 샘플을 확인하는 스크립트"""

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import random

def inspect_current_batches(num_samples=10, skip_samples=0):
    """현재 학습 중인 데이터와 유사한 샘플을 확인"""

    print("=" * 80)
    print("데이터셋 로딩 중...")
    print("=" * 80)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("./kogum-tokenizer")

    # 데이터셋 로드 (학습과 동일한 방식)
    korean_dataset = load_dataset(
        "KORMo-Team/korean-web-collection",
        split="train",
        streaming=True,
    )

    english_dataset = load_dataset(
        "KORMo-Team/dclm-baseline-filtered",
        split="train",
        streaming=True,
    )

    # Interleave (seed는 학습과 동일하게)
    dataset = interleave_datasets(
        [korean_dataset, english_dataset],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    print(f"\n건너뛸 샘플 수: {skip_samples}")
    print(f"확인할 샘플 수: {num_samples}")
    print("\n" + "=" * 80)

    # 특정 위치로 이동 (step 2100 부근 데이터를 보려면)
    # step 2100 * 1024 (batch_size) = 2,150,400 샘플 정도
    iterator = iter(dataset)

    # 건너뛰기
    if skip_samples > 0:
        print(f"샘플 {skip_samples}개 건너뛰는 중...\n")
        for _ in range(skip_samples):
            next(iterator)

    # 샘플 확인
    for i in range(num_samples):
        sample = next(iterator)
        text = sample.get('text', '')

        print(f"\n[샘플 {i+1}/{num_samples}]")
        print(f"길이: {len(text)} 문자")
        print(f"토큰 수: {len(tokenizer.encode(text))}")

        # 언어 추정 (간단한 휴리스틱)
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
        korean_ratio = korean_chars / max(len(text), 1)
        lang = "한국어" if korean_ratio > 0.3 else "영어"
        print(f"언어: {lang} (한글 비율: {korean_ratio:.1%})")

        # 텍스트 품질 체크
        if len(text) < 10:
            print("⚠️  경고: 텍스트가 너무 짧음")

        # 미리보기 (처음 200자)
        preview = text[:200].replace('\n', ' ')
        print(f"미리보기: {preview}...")
        print("-" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="확인할 샘플 수")
    parser.add_argument("--skip", type=int, default=0, help="건너뛸 샘플 수 (step 2100 부근: ~2150000)")
    args = parser.parse_args()

    inspect_current_batches(args.num_samples, args.skip)
