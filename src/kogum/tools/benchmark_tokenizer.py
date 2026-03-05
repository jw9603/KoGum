"""Tokenizer Efficiency Benchmark (K-EXAONE style).

토크나이저의 효율성을 카테고리별로 측정합니다.
- English: 영어 텍스트
- Korean: 한국어 텍스트
- Multilingual: 한영 혼합 텍스트
- STEM: 수학/과학 기술 용어
- Code: 프로그래밍 코드

Chars/Token 비율이 높을수록 효율적인 토크나이저입니다.

Usage:
    python scripts/benchmark_tokenizer.py \
        --tokenizer_path ./kogum-tokenizer \
        --baseline_tokenizer gpt2
"""

import argparse
from typing import Dict, List, Tuple
from transformers import AutoTokenizer


# =============================================================================
# Benchmark Datasets
# =============================================================================

BENCHMARK_TEXTS = {
    "English": [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Natural language processing enables computers to understand human language.",
        "Machine learning algorithms learn patterns from data.",
        "Deep neural networks have revolutionized computer vision.",
    ],
    "Korean": [
        "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        "인공지능은 세상을 변화시키고 있습니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해할 수 있게 합니다.",
        "머신러닝 알고리즘은 데이터로부터 패턴을 학습합니다.",
        "심층 신경망은 컴퓨터 비전을 혁신했습니다.",
    ],
    "Multilingual": [
        "KoGum은 Korean과 English를 동시에 처리할 수 있는 모델입니다.",
        "This model uses Transformer architecture with 0.5B parameters.",
        "데이터셋은 70% 한국어와 30% English로 구성됩니다.",
        "We trained a BPE tokenizer with 80k vocabulary size.",
        "사전학습(Pre-training)과 미세조정(Fine-tuning)을 진행합니다.",
    ],
    "STEM": [
        "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a.",
        "Einstein's mass-energy equivalence: E = mc²",
        "DNA는 Adenine, Thymine, Guanine, Cytosine으로 구성됩니다.",
        "뉴턴의 운동 제2법칙: F = ma (힘 = 질량 × 가속도)",
        "The derivative of x² is 2x, and ∫x²dx = x³/3 + C.",
    ],
    "Code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class Transformer(nn.Module):\n    def __init__(self, d_model=512):\n        super().__init__()",
        "for i in range(10):\n    print(f'Iteration {i}')",
        "import torch\nimport torch.nn as nn\nfrom transformers import AutoModel",
        "const greeting = (name) => {\n  console.log(`Hello, ${name}!`);\n};",
    ],
}


# =============================================================================
# Tokenizer Evaluation
# =============================================================================

def evaluate_tokenizer(
    tokenizer: AutoTokenizer,
    texts: List[str],
) -> Tuple[float, int, int]:
    """토크나이저의 효율성을 측정합니다.

    Args:
        tokenizer: 평가할 토크나이저
        texts: 테스트 텍스트 리스트

    Returns:
        (avg_chars_per_token, total_chars, total_tokens)
    """
    total_chars = 0
    total_tokens = 0

    for text in texts:
        # BOS/EOS 제외하고 순수 텍스트만 토큰화
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_chars += len(text)
        total_tokens += len(tokens)

    avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    return avg_chars_per_token, total_chars, total_tokens


def benchmark_tokenizer(
    tokenizer_path: str,
    baseline_tokenizer: str = None,
) -> Dict[str, Dict[str, float]]:
    """토크나이저를 벤치마크합니다.

    Args:
        tokenizer_path: 평가할 토크나이저 경로
        baseline_tokenizer: 비교 대상 토크나이저 (선택)

    Returns:
        카테고리별 결과 딕셔너리
    """
    print("=" * 70)
    print("Tokenizer Efficiency Benchmark")
    print("=" * 70)

    # 평가 대상 토크나이저 로드
    print(f"\nLoading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # 베이스라인 토크나이저 로드 (선택)
    baseline = None
    if baseline_tokenizer:
        print(f"\nLoading baseline: {baseline_tokenizer}")
        baseline = AutoTokenizer.from_pretrained(baseline_tokenizer)
        print(f"  Vocab size: {baseline.vocab_size:,}")

    # 카테고리별 평가
    results = {}
    print("\n" + "=" * 70)
    print("Results (Chars/Token - higher is better)")
    print("=" * 70)

    for category, texts in BENCHMARK_TEXTS.items():
        # 평가 대상 토크나이저
        avg_cpt, total_chars, total_tokens = evaluate_tokenizer(tokenizer, texts)

        result = {
            "chars_per_token": avg_cpt,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
        }

        # 베이스라인과 비교
        if baseline:
            baseline_cpt, _, baseline_tokens = evaluate_tokenizer(baseline, texts)
            improvement = ((avg_cpt - baseline_cpt) / baseline_cpt) * 100
            result["baseline_cpt"] = baseline_cpt
            result["improvement"] = improvement

            print(f"\n{category:>15}: {avg_cpt:.2f} chars/token "
                  f"(baseline: {baseline_cpt:.2f}, {improvement:+.1f}%)")
        else:
            print(f"\n{category:>15}: {avg_cpt:.2f} chars/token "
                  f"({total_tokens} tokens for {total_chars} chars)")

        results[category] = result

    # 전체 평균
    print("\n" + "=" * 70)
    avg_overall = sum(r["chars_per_token"] for r in results.values()) / len(results)
    print(f"{'Overall Average':>15}: {avg_overall:.2f} chars/token")

    if baseline:
        avg_baseline = sum(r["baseline_cpt"] for r in results.values()) / len(results)
        avg_improvement = ((avg_overall - avg_baseline) / avg_baseline) * 100
        print(f"{'Baseline Avg':>15}: {avg_baseline:.2f} chars/token")
        print(f"{'Improvement':>15}: {avg_improvement:+.1f}%")

    print("=" * 70)

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark tokenizer efficiency (K-EXAONE style)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="평가할 토크나이저 경로",
    )
    parser.add_argument(
        "--baseline_tokenizer",
        type=str,
        default=None,
        help="비교 대상 토크나이저 (예: gpt2, klue/roberta-base)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_tokenizer(
        tokenizer_path=args.tokenizer_path,
        baseline_tokenizer=args.baseline_tokenizer,
    )


if __name__ == "__main__":
    main()
