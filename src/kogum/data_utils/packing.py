"""Sequence packing utilities for efficient training.

Sequence packing은 여러 문서를 하나의 고정 길이 시퀀스로 묶어서
패딩을 제거하고 GPU 활용률을 최대화하는 기법입니다.

예시:
    Doc1: [1, 2, 3, <EOS>]
    Doc2: [4, 5, <EOS>]
    Doc3: [6, 7, 8, 9, <EOS>]

    Packed (seq_len=8):
    [1, 2, 3, <EOS>, 4, 5, <EOS>, 6]
    [7, 8, 9, <EOS>, ...]
"""

from itertools import chain
from typing import Dict, List, Optional

from datasets import Dataset


def pack_sequences(
    examples: Dict[str, List],
    seq_len: int,
) -> Dict[str, List]:
    """여러 시퀀스를 고정 길이 청크로 패킹합니다.

    Args:
        examples: "input_ids" 필드를 포함한 배치 예제
        seq_len: 목표 시퀀스 길이

    Returns:
        패킹된 "input_ids"를 포함한 딕셔너리

    예시:
        Input: {"input_ids": [[1,2,3], [4,5], [6,7,8,9]]}
        seq_len: 5
        Output: {"input_ids": [[1,2,3,4,5], [6,7,8,9,...]]}
    """
    # 모든 input_ids를 하나의 긴 리스트로 평탄화
    # itertools.chain.from_iterable: [[1,2], [3,4]] → [1,2,3,4]
    all_ids = list(chain.from_iterable(examples["input_ids"]))

    # 고정 길이 청크로 분할
    # 나머지는 버림 (마지막 불완전한 청크)
    n_chunks = len(all_ids) // seq_len
    chunks = [all_ids[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)]

    return {"input_ids": chunks}


def pack_sequences_with_boundaries(
    examples: Dict[str, List],
    seq_len: int,
    eos_token_id: int,
) -> Dict[str, List]:
    """문서 경계를 추적하면서 시퀀스를 패킹합니다.

    Intra-document attention masking에 사용됩니다.
    문서 간 attention을 방지하기 위해 각 토큰이 어느 문서에 속하는지 기록합니다.

    Args:
        examples: "input_ids" 필드를 포함한 배치 예제
        seq_len: 목표 시퀀스 길이
        eos_token_id: EOS 토큰 ID (문서 경계 감지용)

    Returns:
        "input_ids"와 "document_ids"를 포함한 딕셔너리

    예시:
        Input: {"input_ids": [[1,2,<EOS>], [4,5,<EOS>]]}
        Output: {
            "input_ids": [[1,2,<EOS>,4,5,<EOS>]],
            "document_ids": [[0,0,0,1,1,1]]  # 문서 ID
        }
    """
    packed_input_ids = []
    packed_document_ids = []

    current_chunk = []
    current_doc_ids = []
    current_doc_id = 0

    # 모든 input_ids를 순회
    for input_ids in examples["input_ids"]:
        for token_id in input_ids:
            current_chunk.append(token_id)
            current_doc_ids.append(current_doc_id)

            # 청크가 목표 길이에 도달하면 저장
            if len(current_chunk) == seq_len:
                packed_input_ids.append(current_chunk)
                packed_document_ids.append(current_doc_ids)
                current_chunk = []
                current_doc_ids = []

            # EOS 토큰을 만나면 문서 ID 증가
            # 다음 토큰부터는 새로운 문서로 간주
            if token_id == eos_token_id:
                current_doc_id += 1

    # 마지막 불완전한 청크는 버림 (표준 관행)
    # 패딩 없이 균일한 길이 유지

    return {
        "input_ids": packed_input_ids,
        "document_ids": packed_document_ids,
    }


def pack_dataset(
    dataset: Dataset,
    seq_len: int,
    num_proc: int = 8,
    batch_size: int = 10000,
    with_boundaries: bool = False,
    eos_token_id: Optional[int] = None,
) -> Dataset:
    """전체 데이터셋을 고정 길이 시퀀스로 패킹합니다.

    Args:
        dataset: "input_ids" 컬럼을 포함한 HuggingFace Dataset
        seq_len: 목표 시퀀스 길이 (예: 2048, 4096)
        num_proc: 병렬 처리에 사용할 프로세스 수
        batch_size: map 연산의 배치 크기 (메모리 vs 속도 트레이드오프)
        with_boundaries: 문서 경계 추적 여부
        eos_token_id: EOS 토큰 ID (with_boundaries=True일 때 필수)

    Returns:
        패킹된 Dataset

    사용 예시:
        >>> from datasets import load_dataset
        >>> ds = load_dataset("...")
        >>> packed_ds = pack_dataset(ds, seq_len=2048)
        >>> # 이제 각 샘플이 정확히 2048 토큰
    """
    # IterableDataset 여부 확인
    from datasets import IterableDataset
    is_iterable = isinstance(dataset, IterableDataset)

    # map 파라미터 준비
    map_kwargs = {
        "batched": True,
        "batch_size": batch_size,
        "remove_columns": dataset.column_names,
    }

    # IterableDataset은 num_proc을 지원하지 않음
    if not is_iterable and num_proc is not None:
        map_kwargs["num_proc"] = num_proc
        map_kwargs["desc"] = f"Packing sequences (seq_len={seq_len})"

    if with_boundaries:
        if eos_token_id is None:
            raise ValueError("with_boundaries=True일 때 eos_token_id 필수")

        return dataset.map(
            pack_sequences_with_boundaries,
            fn_kwargs={"seq_len": seq_len, "eos_token_id": eos_token_id},
            **map_kwargs,
        )
    else:
        return dataset.map(
            pack_sequences,
            fn_kwargs={"seq_len": seq_len},
            **map_kwargs,
        )


def tokenize_and_pack(
    dataset: Dataset,
    tokenizer,
    seq_len: int,
    text_column: str = "text",
    num_proc: int = 8,
    batch_size: int = 1000,
    add_eos: bool = True,
) -> Dataset:
    """Raw 텍스트를 토큰화하고 패킹까지 한 번에 처리합니다.

    편의 함수: tokenization + packing을 결합

    Args:
        dataset: 텍스트 컬럼을 포함한 Dataset
        tokenizer: HuggingFace tokenizer
        seq_len: 목표 시퀀스 길이
        text_column: 텍스트 컬럼 이름
        num_proc: 병렬 처리 프로세스 수
        batch_size: 토큰화 배치 크기
        add_eos: 각 문서 끝에 EOS 토큰 추가 여부

    Returns:
        토큰화 및 패킹된 Dataset

    사용 예시:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("./kogum-tokenizer")
        >>> ds = load_dataset("...")
        >>> packed_ds = tokenize_and_pack(ds, tokenizer, seq_len=2048)
    """

    def tokenize_fn(examples):
        """배치 단위로 텍스트를 토큰화합니다."""
        texts = examples[text_column]

        # 토큰화 (special tokens 추가 안 함)
        # add_special_tokens=False: BOS/EOS를 자동 추가하지 않음
        #   - Pre-training에서는 데이터 전처리 단계에서 수동으로 EOS 추가
        #   - 이렇게 하면 sequence packing 시 문서 경계 명확
        tokenized = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,  # 패킹 후 attention mask는 불필요
        )

        # 각 문서 끝에 EOS 토큰 추가
        # 문서 경계 표시용
        if add_eos and tokenizer.eos_token_id is not None:
            tokenized["input_ids"] = [
                ids + [tokenizer.eos_token_id] for ids in tokenized["input_ids"]
            ]

        return tokenized

    # Step 1: 토큰화
    # IterableDataset 여부 확인
    from datasets import IterableDataset
    is_iterable = isinstance(dataset, IterableDataset)

    if is_iterable:
        print(f"Tokenizing samples (streaming mode)...")
        # IterableDataset은 num_proc을 지원하지 않음
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
        )
    else:
        print(f"Tokenizing {len(dataset):,} samples...")
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Tokenizing",
        )

    # Step 2: 패킹
    print(f"Packing into seq_len={seq_len}...")
    if is_iterable:
        # IterableDataset은 num_proc을 지원하지 않음
        packed_dataset = pack_dataset(
            tokenized_dataset,
            seq_len=seq_len,
            num_proc=None,
            batch_size=batch_size * 10,
        )
    else:
        packed_dataset = pack_dataset(
            tokenized_dataset,
            seq_len=seq_len,
            num_proc=num_proc,
            batch_size=batch_size * 10,
        )

    # 결과 출력
    if is_iterable:
        print(f"Packed dataset ready (streaming mode)")
    else:
        print(f"Packed dataset size: {len(packed_dataset):,} sequences")

    return packed_dataset
