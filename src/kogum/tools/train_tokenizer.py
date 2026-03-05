"""KoGum Tokenizer Training Script.

BPE(Byte Pair Encoding) 토크나이저를 한국어-영어 혼합 코퍼스로 학습합니다.
Special tokens는 KORMo 스타일로 vocabulary 끝에 배치됩니다.

=== BPE 알고리즘 원리 ===
1. 모든 텍스트를 바이트(UTF-8) 단위로 분리
2. 가장 자주 등장하는 바이트 쌍을 찾음
3. 해당 쌍을 새로운 토큰으로 병합
4. vocab_size에 도달할 때까지 2-3 반복

예시:
  "안녕" → ['ì', '•', '?', 'ë', ...] (바이트)
  → 자주 나오는 쌍 병합
  → ['안', '녕'] 또는 ['안녕']

Usage:
    python scripts/train_tokenizer.py \
        --output_dir ./kogum-tokenizer \
        --vocab_size 80000 \
        --ko_ratio 0.7 \
        --sample_size 10000000

    # With streaming (for large datasets)
    python scripts/train_tokenizer.py \
        --output_dir ./kogum-tokenizer \
        --streaming \
        --interleave
"""

import argparse
import random
from pathlib import Path
from typing import Iterator, List, Optional

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC
from transformers import PreTrainedTokenizerFast


# =============================================================================
# Special Tokens 정의
# =============================================================================
# KORMo 스타일: vocabulary 끝에 배치
# 순서가 중요함 - 이 순서대로 ID가 부여됨
SPECIAL_TOKENS = [
    "<|PAD|>",   # 패딩 토큰: 배치 내 길이 맞춤용 (ID: vocab_base + 0)
    "<|BOS|>",   # 시퀀스 시작 토큰 (ID: vocab_base + 1)
    "<|EOS|>",   # 시퀀스 끝 토큰 (ID: vocab_base + 2)
    "<|UNK|>",   # 미등록 토큰: vocab에 없는 토큰 대체 (ID: vocab_base + 3)
]

# SFT/Chat 단계에서 사용할 추가 토큰 (미리 예약)
ADDITIONAL_TOKENS = [
    "<|BOT|>",           # 봇(어시스턴트) 발화 시작
    "<|EOT|>",           # 턴 종료 (End of Turn)
    "<think>",           # 추론/사고 과정 시작 (Chain-of-Thought)
    "</think>",          # 추론/사고 과정 끝
    "<tool_call>",       # 도구 호출 시작
    "</tool_call>",      # 도구 호출 끝
    "<tool_response>",   # 도구 응답 시작
    "</tool_response>",  # 도구 응답 끝
]

# 연속 개행 토큰 (KORMo 스타일)
# 코드나 문서에서 여러 줄 개행을 효율적으로 표현
# 즉, \n\n 부터 \n×31 을 단일 토큰으로 처리
NEWLINE_TOKENS = ["\n" * i for i in range(2, 32)]  # \n\n ~ \n×31


# =============================================================================
# CLI 인자 파싱
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train KoGum tokenizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kogum-tokenizer",
        help="토크나이저 저장 경로",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=80000,
        help="목표 vocabulary 크기 (special tokens 제외)",
    )
    parser.add_argument(
        "--ko_ratio",
        type=float,
        default=0.7,
        help="한국어 데이터 비율 (0.0-1.0)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10_000_000,
        help="학습에 사용할 총 샘플 수",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="토큰이 vocab에 포함되기 위한 최소 등장 횟수",
    )
    parser.add_argument(
        "--ko_dataset",
        type=str,
        default="KORMo-Team/korean-web-collection",
        help="한국어 데이터셋 이름 (HuggingFace)",
    )
    parser.add_argument(
        "--en_dataset",
        type=str,
        default="KORMo-Team/dclm-baseline-filtered",
        help="영어 데이터셋 이름 (HuggingFace)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="텍스트가 담긴 컬럼 이름",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="스트리밍 모드 사용 (대용량 데이터셋 권장)",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="한국어/영어 데이터를 섞어서 학습 (vocab 균형 향상)",
    )
    parser.add_argument(
        "--no_post_processor",
        action="store_true",
        help="BOS/EOS 자동 추가 비활성화 (사전학습용 권장)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드",
    )
    return parser.parse_args()


# =============================================================================
# 데이터 로드 및 샘플링
# =============================================================================
def load_and_sample_data(
    ko_dataset: str,
    en_dataset: str,
    ko_ratio: float,
    sample_size: int,
    text_column: str,
    streaming: bool,
    interleave: bool,
    seed: int,
) -> Iterator[str]:
    """한국어/영어 데이터셋에서 데이터를 로드하고 샘플링합니다.

    Args:
        ko_dataset: 한국어 데이터셋 이름
        en_dataset: 영어 데이터셋 이름
        ko_ratio: 한국어 비율 (0.7 = 70%)
        sample_size: 총 샘플 수
        text_column: 텍스트 컬럼 이름
        streaming: 스트리밍 모드 여부
        interleave: 데이터 섞기 여부
        seed: 랜덤 시드

    Yields:
        텍스트 문자열
    """
    # 언어별 샘플 수 계산
    # 예: 10M 샘플, 70% 한국어 → 한국어 7M, 영어 3M
    ko_samples = int(sample_size * ko_ratio)
    en_samples = sample_size - ko_samples

    print(f"Loading datasets...")
    print(f"  Korean: {ko_dataset} ({ko_samples:,} samples)")
    print(f"  English: {en_dataset} ({en_samples:,} samples)")
    print(f"  Interleave: {interleave}")

    random.seed(seed)

    # =========================================================================
    # 한국어 데이터 로드
    # =========================================================================
    print("\nLoading Korean dataset...")
    ko_texts = []
    try:
        # HuggingFace datasets에서 로드
        # streaming=True면 전체 다운로드 없이 필요한 만큼만 가져옴
        ko_ds = load_dataset(ko_dataset, split="train", streaming=streaming)

        if streaming:
            # 스트리밍 모드: 버퍼 내에서 셔플 (메모리 효율적)
            ko_ds = ko_ds.shuffle(seed=seed, buffer_size=10000)
            for i, item in enumerate(ko_ds):
                if i >= ko_samples:
                    break
                text = item.get(text_column, "")
                # 빈 텍스트 필터링
                if text and len(text.strip()) > 0:
                    ko_texts.append(text)
                if (i + 1) % 100000 == 0:
                    print(f"  Korean: {i + 1:,} samples loaded...")
        else:
            # 일반 모드: 전체 로드 후 셔플
            ko_ds = ko_ds.shuffle(seed=seed)
            for i, item in enumerate(ko_ds):
                if i >= ko_samples:
                    break
                text = item.get(text_column, "")
                if text and len(text.strip()) > 0:
                    ko_texts.append(text)
    except Exception as e:
        print(f"Warning: Failed to load Korean dataset: {e}")
        print("Continuing with empty Korean data")

    print(f"  Korean samples loaded: {len(ko_texts):,}")

    # =========================================================================
    # 영어 데이터 로드
    # =========================================================================
    print("\nLoading English dataset...")
    en_texts = []
    try:
        en_ds = load_dataset(en_dataset, split="train", streaming=streaming)
        if streaming:
            en_ds = en_ds.shuffle(seed=seed, buffer_size=10000)
            for i, item in enumerate(en_ds):
                if i >= en_samples:
                    break
                text = item.get(text_column, "")
                if text and len(text.strip()) > 0:
                    en_texts.append(text)
                if (i + 1) % 100000 == 0:
                    print(f"  English: {i + 1:,} samples loaded...")
        else:
            en_ds = en_ds.shuffle(seed=seed)
            for i, item in enumerate(en_ds):
                if i >= en_samples:
                    break
                text = item.get(text_column, "")
                if text and len(text.strip()) > 0:
                    en_texts.append(text)
    except Exception as e:
        print(f"Warning: Failed to load English dataset: {e}")
        print("Continuing with empty English data")

    print(f"  English samples loaded: {len(en_texts):,}")
    print(f"\nTotal samples: {len(ko_texts) + len(en_texts):,}")

    # =========================================================================
    # 데이터 반환 (인터리빙 또는 순차)
    # =========================================================================
    if interleave:
        # 인터리빙: 한국어와 영어를 비율에 맞게 섞음
        # 이렇게 하면 BPE 학습 시 두 언어의 vocab이 균형있게 형성됨
        #
        # 예: ko_ratio=0.7이면
        #   한국어 7개 → 영어 3개 → 한국어 7개 → 영어 3개 → ...
        print("Interleaving data...")
        ko_idx, en_idx = 0, 0

        all_texts = []
        while ko_idx < len(ko_texts) or en_idx < len(en_texts):
            # 비율에 맞게 한국어 추가 (70% → 7개씩)
            for _ in range(int(ko_ratio * 10)):
                if ko_idx < len(ko_texts):
                    all_texts.append(ko_texts[ko_idx])
                    ko_idx += 1
            # 비율에 맞게 영어 추가 (30% → 3개씩)
            for _ in range(int((1 - ko_ratio) * 10)):
                if en_idx < len(en_texts):
                    all_texts.append(en_texts[en_idx])
                    en_idx += 1

        for text in all_texts:
            yield text
    else:
        # 순차: 한국어 전부 → 영어 전부
        for text in ko_texts:
            yield text
        for text in en_texts:
            yield text


# =============================================================================
# BPE 토크나이저 학습
# =============================================================================
def train_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int,
    min_frequency: int,
) -> Tokenizer:
    """BPE 토크나이저를 학습합니다.

    === BPE 학습 과정 ===
    1. 텍스트를 바이트 단위로 분리 (UTF-8)
    2. 전체 코퍼스에서 바이트 쌍 빈도 계산
    3. 가장 빈번한 쌍을 새 토큰으로 병합
    4. vocab_size에 도달할 때까지 반복

    Args:
        text_iterator: 텍스트 이터레이터
        vocab_size: 목표 vocabulary 크기
        min_frequency: 최소 등장 횟수

    Returns:
        학습된 Tokenizer 객체
    """
    print(f"\nTraining BPE tokenizer...")
    print(f"  Target vocab size: {vocab_size:,}")
    print(f"  Min frequency: {min_frequency}")

    # =========================================================================
    # Step 1: BPE 모델 초기화
    # =========================================================================
    # BPE = Byte Pair Encoding
    # unk_token: vocabulary에 없는 바이트 시퀀스를 대체할 토큰
    tokenizer = Tokenizer(models.BPE(unk_token="<|UNK|>"))

    # =========================================================================
    # Step 2: Normalizer 설정
    # =========================================================================
    # NFKC: 유니코드 정규화
    # - 호환 문자를 표준 형태로 변환
    # - 예: ＡＢＣ → ABC, ① → 1
    # - 한국어 자모 분리/조합 일관성 유지
    tokenizer.normalizer = NFKC()

    # =========================================================================
    # Step 3: Pre-tokenizer 설정
    # =========================================================================
    # ByteLevel: 텍스트를 UTF-8 바이트로 변환
    # - 모든 유니코드 문자 처리 가능
    # - OOV(Out-of-Vocabulary) 문제 해결
    # - GPT-2, LLaMA, KORMo가 사용하는 방식
    #
    # 예: "안녕" → UTF-8 바이트 → BPE 처리
    #
    # add_prefix_space=False: 단어 앞에 공백 추가 안 함
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # =========================================================================
    # Step 4: Decoder 설정
    # =========================================================================
    # ByteLevel decoder: 토큰 ID → 바이트 → 텍스트 복원
    tokenizer.decoder = decoders.ByteLevel()

    # =========================================================================
    # Step 5: Trainer 설정
    # =========================================================================
    # BpeTrainer: BPE 알고리즘으로 vocabulary 학습
    #
    # vocab_size: 최종 vocabulary 크기
    # min_frequency: 이 횟수 미만으로 등장한 쌍은 병합 안 함
    #               (노이즈 제거, 희귀 토큰 방지)
    # special_tokens: 학습 전에 미리 추가할 특수 토큰
    # show_progress: 진행 상황 표시
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # =========================================================================
    # Step 6: 텍스트 수집
    # =========================================================================
    # tokenizers 라이브러리는 리스트 입력 필요
    # (이터레이터를 리스트로 변환)
    print("Collecting texts for training...")
    texts = list(text_iterator)
    print(f"  Collected {len(texts):,} texts")

    # =========================================================================
    # Step 7: BPE 학습 실행
    # =========================================================================
    # 이 단계에서 실제 BPE 알고리즘이 실행됨:
    # 1. 모든 텍스트를 바이트로 변환
    # 2. 바이트 쌍 빈도 계산
    # 3. 가장 빈번한 쌍 병합 → 새 토큰
    # 4. vocab_size 도달까지 반복
    #
    # 시간이 오래 걸리는 구간 (CPU bound)
    print("Starting BPE training...")
    print("  (이 과정은 수 시간 소요될 수 있습니다...)")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    print(f"  Trained vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


# =============================================================================
# Special Tokens 추가
# =============================================================================
def add_special_tokens(tokenizer: Tokenizer, no_post_processor: bool) -> Tokenizer:
    """추가 토큰을 등록하고 post-processor를 설정합니다.

    Args:
        tokenizer: 학습된 토크나이저
        no_post_processor: True면 BOS/EOS 자동 추가 비활성화

    Returns:
        설정이 완료된 토크나이저
    """
    vocab_size_before = tokenizer.get_vocab_size()

    # =========================================================================
    # SFT용 추가 토큰 등록
    # =========================================================================
    # 미리 등록해두면 나중에 SFT할 때 vocabulary 변경 없이 사용 가능
    for token in ADDITIONAL_TOKENS:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token])

    # =========================================================================
    # 연속 개행 토큰 등록
    # =========================================================================
    # \n\n, \n\n\n, ... 을 단일 토큰으로 처리
    # 코드나 마크다운에서 효율적
    for token in NEWLINE_TOKENS:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token])

    print(f"\nAdded {tokenizer.get_vocab_size() - vocab_size_before} additional tokens")

    # =========================================================================
    # Post-processor 설정 (선택적)
    # =========================================================================
    # Post-processor: 토큰화 후 BOS/EOS 자동 추가
    #
    # 사전학습(Pre-training)에서는 보통 비활성화:
    #   - 데이터 전처리 단계에서 직접 EOS 추가
    #   - Sequence packing 시 문서 경계 표시용
    #
    # SFT/추론에서는 활성화:
    #   - 자동으로 <|BOS|> text <|EOS|> 형태로 변환
    if not no_post_processor:
        bos_id = tokenizer.token_to_id("<|BOS|>")
        eos_id = tokenizer.token_to_id("<|EOS|>")

        # TemplateProcessing: 토큰화 결과에 템플릿 적용
        # single: 단일 텍스트용 템플릿
        # pair: 텍스트 쌍용 템플릿 (예: QA)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<|BOS|>:0 $A:0 <|EOS|>:0",
            pair=f"<|BOS|>:0 $A:0 <|EOS|>:0 <|BOS|>:1 $B:1 <|EOS|>:1",
            special_tokens=[
                ("<|BOS|>", bos_id),
                ("<|EOS|>", eos_id),
            ],
        )
        print("Post-processor enabled (auto BOS/EOS)")
    else:
        print("Post-processor disabled (manual BOS/EOS)")

    print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


# =============================================================================
# 토크나이저 저장
# =============================================================================
def save_tokenizer(tokenizer: Tokenizer, output_dir: str):
    """토크나이저를 HuggingFace 형식으로 저장합니다.

    저장되는 파일:
    - tokenizer.json: 전체 토크나이저 설정
    - tokenizer_config.json: HuggingFace 호환 설정
    - special_tokens_map.json: 특수 토큰 매핑
    - CONFIG_SUMMARY.md: 모델 config 업데이트용 요약
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Special token ID 조회
    pad_id = tokenizer.token_to_id("<|PAD|>")
    bos_id = tokenizer.token_to_id("<|BOS|>")
    eos_id = tokenizer.token_to_id("<|EOS|>")
    unk_id = tokenizer.token_to_id("<|UNK|>")

    # =========================================================================
    # HuggingFace 형식으로 래핑
    # =========================================================================
    # PreTrainedTokenizerFast: HuggingFace transformers와 호환되는 래퍼
    # 이렇게 하면 AutoTokenizer.from_pretrained()로 로드 가능
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|BOS|>",
        eos_token="<|EOS|>",
        unk_token="<|UNK|>",
        pad_token="<|PAD|>",
        clean_up_tokenization_spaces=False,
    )

    # 저장
    hf_tokenizer.save_pretrained(output_dir)

    print(f"\nTokenizer saved to: {output_dir}")
    print(f"\nSpecial token IDs:")
    print(f"  <|PAD|>: {pad_id}")
    print(f"  <|BOS|>: {bos_id}")
    print(f"  <|EOS|>: {eos_id}")
    print(f"  <|UNK|>: {unk_id}")

    # =========================================================================
    # Config 요약 저장
    # =========================================================================
    # 모델 config 업데이트에 필요한 정보를 마크다운으로 저장
    vocab_size = tokenizer.get_vocab_size()
    config_summary = f"""# KoGum Tokenizer Configuration

## Vocabulary
vocab_size: {vocab_size}

## Special Token IDs (for model config)
pad_token_id: {pad_id}
bos_token_id: {bos_id}
eos_token_id: {eos_id}
unk_token_id: {unk_id}

## Update your model config (kogum_0.5B.yaml)

```yaml
vocab_size: {vocab_size}
pad_token_id: {pad_id}
bos_token_id: {bos_id}
eos_token_id: {eos_id}
```

## Python code to update config

```python
from kogum import KoGumConfig

config = KoGumConfig(
    vocab_size={vocab_size},
    pad_token_id={pad_id},
    bos_token_id={bos_id},
    eos_token_id={eos_id},
    # ... other params
)
```
"""

    with open(output_path / "CONFIG_SUMMARY.md", "w") as f:
        f.write(config_summary)

    print(f"\nConfig summary saved to: {output_path / 'CONFIG_SUMMARY.md'}")


# =============================================================================
# 토크나이저 테스트
# =============================================================================
def test_tokenizer(output_dir: str):
    """학습된 토크나이저를 테스트합니다.

    한국어, 영어, 혼합 텍스트로 토큰화 품질을 확인합니다.
    Chars/Token 비율이 높을수록 효율적인 토크나이저입니다.
    """
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("Testing tokenizer...")
    print("=" * 60)

    # HuggingFace 방식으로 로드 (저장이 제대로 됐는지 확인)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)

    # 테스트 텍스트 (다양한 케이스)
    test_texts = [
        "안녕하세요, KoGum 모델입니다.",           # 한국어
        "Hello, this is KoGum model.",            # 영어
        "한국어와 English를 함께 처리합니다.",      # 혼합
        "The quick brown fox jumps over the lazy dog.",  # 영어 팬그램
        "인공지능(AI)은 미래 기술의 핵심입니다.",    # 한국어 + 약어
        "def hello_world():\n    print('Hello, World!')",  # 코드
        "서울특별시 강남구 테헤란로 123번길",       # 주소
    ]

    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    for text in test_texts:
        # 토큰화
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)

        print(f"\n[Input]: {text}")
        print(f"[Tokens]: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"[IDs]: {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"[Decoded]: {decoded}")
        # Chars/Token: 높을수록 좋음 (적은 토큰으로 많은 문자 표현)
        print(f"[Token count]: {len(tokens)} | [Chars/Token]: {len(text)/len(tokens):.2f}")


# =============================================================================
# 메인 함수
# =============================================================================
def main():
    args = parse_args()

    print("=" * 60)
    print("KoGum Tokenizer Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Output: {args.output_dir}")
    print(f"  Vocab size: {args.vocab_size:,}")
    print(f"  Ko:En ratio: {args.ko_ratio:.0%}:{1-args.ko_ratio:.0%}")
    print(f"  Sample size: {args.sample_size:,}")
    print(f"  Min frequency: {args.min_frequency}")
    print(f"  Streaming: {args.streaming}")
    print(f"  Interleave: {args.interleave}")
    print(f"  Post-processor: {'disabled' if args.no_post_processor else 'enabled'}")

    # Step 1: 데이터 로드 및 샘플링
    text_iterator = load_and_sample_data(
        ko_dataset=args.ko_dataset,
        en_dataset=args.en_dataset,
        ko_ratio=args.ko_ratio,
        sample_size=args.sample_size,
        text_column=args.text_column,
        streaming=args.streaming,
        interleave=args.interleave,
        seed=args.seed,
    )

    # Step 2: BPE 토크나이저 학습
    tokenizer = train_tokenizer(
        text_iterator=text_iterator,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Step 3: Special tokens 추가
    tokenizer = add_special_tokens(tokenizer, args.no_post_processor)

    # Step 4: 저장
    save_tokenizer(tokenizer, args.output_dir)

    # Step 5: 테스트
    test_tokenizer(args.output_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
