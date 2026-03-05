# -*- coding: utf-8 -*-
"""체크포인트 평가 스크립트

간단한 텍스트 생성으로 모델 품질을 확인합니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer

from kogum.model import KoGumForCausalLM


def evaluate_checkpoint(checkpoint_path: str, prompts: list[str], max_new_tokens: int = 100):
    """체크포인트를 로드하고 텍스트 생성 테스트"""

    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 80)

    # 모델과 토크나이저 로드
    model = KoGumForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("./kogum-tokenizer")

    # GPU 사용 가능하면 GPU로
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 80)
    print()

    # 각 프롬프트에 대해 생성
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        print("-" * 80)

        # 토크나이즈
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Generated:\n{generated_text}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="체크포인트 경로")
    parser.add_argument("--max_tokens", type=int, default=100, help="생성할 최대 토큰 수")
    args = parser.parse_args()

    # 테스트 프롬프트
    prompts = [
        "인공지능은",
        "서울은 대한민국의",
        "Once upon a time",
        "Python is a",
        "오늘 날씨는",
    ]

    evaluate_checkpoint(args.checkpoint, prompts, args.max_tokens)
