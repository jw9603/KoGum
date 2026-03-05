# -*- coding: utf-8 -*-
"""KoGum SFT (Supervised Fine-Tuning) Script.

Mid-training된 KoGum 모델을 instruction-following 데이터로 fine-tuning합니다.
- 한국어 2개 + 영어 2개 데이터셋
- Non-streaming 모드
- Chat template 기반 tokenization
- <think> reasoning + <tool_call> tool calling 지원
- 대용량 데이터셋은 샘플링하여 총 ~3M samples

Usage:
    torchrun --nproc_per_node=2 -m kogum.train.sft \
        --model_name_or_path ./kogum-0.5B-16k-midtrain/final_model
"""

import os
import warnings
import torch
import transformers
from datasets import load_dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer, set_seed

warnings.filterwarnings("ignore", message="cuDNN SDPA backward got grad_output.strides")

from kogum.model import KoGumConfig, KoGumForCausalLM
from kogum.train import (
    KoGumTrainer,
    KoGumTrainingArguments,
)


def print_rank0(*args, **kwargs):
    """분산 학습 시 메인 프로세스(rank 0)에서만 출력합니다."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def sample_dataset(ds, max_samples, seed):
    """데이터셋이 max_samples보다 크면 셔플 후 샘플링합니다."""
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds


# 모든 데이터셋에 통일할 스키마 (role + content만)
MESSAGES_FEATURES = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}]
})


def _clean_messages(example_messages):
    """messages에서 role/content만 추출하여 스키마를 통일합니다."""
    return [{"role": m["role"], "content": m.get("content") or ""} for m in example_messages]


def load_sft_datasets(
    tokenizer,
    max_length: int,
    seed: int,
    nemo_ko_samples: int = 200000,
    if_bilingual_samples: int = 80000,
    nemotron_chat_samples: int = 30000,
    nemotron_code_samples: int = 20000,
    nemotron_math_samples: int = 20000,
    nemotron_tool_samples: int = 15000,
    smoltalk_samples: int = 35000,
):
    """SFT 데이터셋을 로드하고 chat template으로 tokenize합니다.

    한국어 (~280K):
        - KORMo-Team/NemoPost-ko-synth-sft (200K 샘플링)
        - KORMo-Team/IF-bilingual-sft (80K 샘플링)

    영어 (~120K):
        - nvidia/Nemotron-Post-Training-Dataset-v1 (chat 30K, code 20K, math 20K, tool_calling 15K)
        - HuggingFaceTB/smoltalk2 (35K 샘플링)

    총 약 ~400K samples (한/영 7:3)
    """

    nemotron_sample_map = {
        "chat": nemotron_chat_samples,
        "code": nemotron_code_samples,
        "math": nemotron_math_samples,
        "tool_calling": nemotron_tool_samples,
    }

    # =================================================================
    # 1. KORMo-Team/NemoPost-ko-synth-sft (한국어 QA)
    # =================================================================
    print_rank0("  Loading: KORMo-Team/NemoPost-ko-synth-sft")
    nemo_ko = load_dataset("KORMo-Team/NemoPost-ko-synth-sft", split="train")

    # question/answer → messages 변환
    nemo_ko = nemo_ko.map(
        lambda x: {"messages": [
            {"role": "user", "content": x["question"]},
            {"role": "assistant", "content": x["answer"]},
        ]},
        remove_columns=nemo_ko.column_names,
        features=MESSAGES_FEATURES,
    )
    print_rank0(f"    -> {len(nemo_ko):,} samples (before sampling)")
    nemo_ko = sample_dataset(nemo_ko, nemo_ko_samples, seed)
    print_rank0(f"    -> {len(nemo_ko):,} samples (after sampling)")

    # =================================================================
    # 2. KORMo-Team/IF-bilingual-sft (한국어 conversation)
    # =================================================================
    print_rank0("  Loading: KORMo-Team/IF-bilingual-sft (Korean)")
    ko_subsets = []
    # Korean: Magpie, MathInstruct, Mixture_of_Thoughts_science, OpenHermes2.5, PubMedQA
    # Korean-Reasoning: Magpie, Nemotron_Post_Train_V1
    ko_split_map = {
        "Korean": ["Magpie", "MathInstruct", "Mixture_of_Thoughts_science", "OpenHermes2.5", "PubMedQA"],
        "Korean-Reasoning": ["Magpie", "Nemotron_Post_Train_V1"],
    }
    for subset, splits in ko_split_map.items():
        for split_name in splits:
            try:
                ds = load_dataset("KORMo-Team/IF-bilingual-sft", subset, split=split_name)
                # conversation → messages 변환 (role/content만 추출, reasoning_content 등 제거)
                ds = ds.map(
                    lambda x: {"messages": [{"role": m["role"], "content": m["content"]} for m in x["conversation"]]},
                    remove_columns=ds.column_names,
                    features=MESSAGES_FEATURES,
                )
                ko_subsets.append(ds)
                print_rank0(f"    -> {subset}/{split_name}: {len(ds):,}")
            except Exception as e:
                print_rank0(f"    ! {subset}/{split_name}: skipped ({e})")

    if_bilingual = concatenate_datasets(ko_subsets) if ko_subsets else None
    if if_bilingual:
        print_rank0(f"    -> Total IF-bilingual: {len(if_bilingual):,} (before sampling)")
        if_bilingual = sample_dataset(if_bilingual, if_bilingual_samples, seed)
        print_rank0(f"    -> Total IF-bilingual: {len(if_bilingual):,} (after sampling)")

    # =================================================================
    # 3. nvidia/Nemotron-Post-Training-Dataset-v1 (영어)
    #    chat(30K), code(20K), math(20K), tool_calling(15K)
    # =================================================================
    print_rank0("  Loading: nvidia/Nemotron-Post-Training-Dataset-v1")
    nemotron_splits = []
    import json as _json

    for split_name in ["chat", "code", "math", "tool_calling"]:
        try:
            ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split=split_name)

            # tool_calling split: tool_calls를 텍스트로 변환하여 content에 포함
            if split_name == "tool_calling":
                def convert_nemotron_tool(example):
                    messages = []
                    for m in example["messages"]:
                        content = m.get("content") or ""
                        role = m["role"]

                        if role == "assistant" and m.get("tool_calls"):
                            tool_call_texts = []
                            for tc in m["tool_calls"]:
                                func = tc.get("function", tc) if isinstance(tc, dict) else tc
                                if isinstance(func, dict):
                                    name = func.get("name", "")
                                    args = func.get("arguments", {})
                                    args_str = args if isinstance(args, str) else _json.dumps(args, ensure_ascii=False)
                                    tool_call_texts.append(
                                        f'<tool_call>\n{{"name": "{name}", "arguments": {args_str}}}\n</tool_call>'
                                    )
                            if tool_call_texts:
                                content = content + "\n" + "\n".join(tool_call_texts) if content else "\n".join(tool_call_texts)

                        if role == "tool":
                            role = "user"
                            content = f"<tool_response>\n{content}\n</tool_response>"

                        messages.append({"role": role, "content": content})
                    return {"messages": messages}

                ds = ds.map(convert_nemotron_tool, remove_columns=ds.column_names, features=MESSAGES_FEATURES)
            else:
                ds = ds.map(
                    lambda x: {"messages": _clean_messages(x["messages"])},
                    remove_columns=ds.column_names,
                    features=MESSAGES_FEATURES,
                )

            # 모든 split 샘플링
            max_samples = nemotron_sample_map.get(split_name)
            if max_samples:
                print_rank0(f"    -> {split_name}: {len(ds):,} (before sampling)")
                ds = sample_dataset(ds, max_samples, seed)

            nemotron_splits.append(ds)
            print_rank0(f"    -> {split_name}: {len(ds):,}")
        except Exception as e:
            print_rank0(f"    ! {split_name}: skipped ({e})")

    nemotron = concatenate_datasets(nemotron_splits) if nemotron_splits else None
    if nemotron:
        print_rank0(f"    -> Total Nemotron: {len(nemotron):,} samples")

    # =================================================================
    # 4. HuggingFaceTB/smoltalk2 (영어 SFT - think/no_think splits)
    # =================================================================
    print_rank0("  Loading: HuggingFaceTB/smoltalk2 (SFT)")
    smoltalk_splits = []

    # Think splits (reasoning 포함)
    think_splits = [
        "OpenThoughts3_1.2M_think",
        "smoltalk_multilingual8_Qwen3_32B_think",
        "multi_turn_reasoning_if_think",
        "smoltalk_systemchats_Qwen3_32B_think",
        "aya_dataset_Qwen3_32B_think",
        "table_gpt_Qwen3_32B_think",
        "smolagents_toolcalling_traces_think",
        "smoltalk_everyday_convs_reasoning_Qwen3_32B_think",
        "s1k_1.1_think",
    ]

    # No-think splits (reasoning 없음)
    no_think_splits = [
        "OpenThoughts3_1.2M_no_think_no_think",
        "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "OpenHermes_2.5_no_think",
        "smoltalk_multilingual_8languages_lang_5_no_think",
        "smoltalk_smollm3_smol_summarize_no_think",
        "Mixture_of_Thoughts_science_no_think",
        "xlam_traces_no_think",
        "smoltalk_smollm3_smol_rewrite_no_think",
        "smoltalk_smollm3_systemchats_30k_no_think",
        "smoltalk_smollm3_explore_instruct_rewriting_no_think",
        "tulu_3_sft_personas_instruction_following_no_think",
        "table_gpt_no_think",
        "hermes_function_calling_v1_no_think",
        "smoltalk_smollm3_everyday_conversations_no_think",
    ]

    for split_name in think_splits + no_think_splits:
        try:
            ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split_name)
            # role/content만 추출 (tool_calls 등 extra fields 제거하여 스키마 통일)
            ds = ds.map(
                lambda x: {"messages": _clean_messages(x["messages"])},
                remove_columns=ds.column_names,
                features=MESSAGES_FEATURES,
            )
            smoltalk_splits.append(ds)
            print_rank0(f"    -> SFT/{split_name}: {len(ds):,}")
        except Exception as e:
            print_rank0(f"    ! SFT/{split_name}: skipped ({e})")

    smoltalk = concatenate_datasets(smoltalk_splits) if smoltalk_splits else None
    if smoltalk:
        print_rank0(f"    -> Total smoltalk2: {len(smoltalk):,} (before sampling)")
        smoltalk = sample_dataset(smoltalk, smoltalk_samples, seed)
        print_rank0(f"    -> Total smoltalk2: {len(smoltalk):,} (after sampling)")

    # =================================================================
    # 전체 합치기 + 셔플
    # =================================================================
    all_datasets = []
    for ds in [nemo_ko, if_bilingual, nemotron, smoltalk]:
        if ds is not None:
            all_datasets.append(ds)

    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=seed)
    print_rank0(f"\n  Total SFT samples: {len(combined):,}")

    # =================================================================
    # Chat template으로 tokenize
    # =================================================================
    print_rank0(f"  Tokenizing with chat template (max_length={max_length})...")

    def tokenize_chat(example):
        messages = example["messages"]

        # chat template 적용 (tool_calls는 이미 텍스트로 변환됨)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # labels 생성: user/system 부분은 -100으로 마스킹, assistant 부분만 학습
        input_ids = tokenized["input_ids"]
        labels = [-100] * len(input_ids)

        # <|BOT|>assistant\n 이후부터 <|EOT|>까지를 학습 대상으로 설정
        # assistant 응답 안의 <think>...</think>, <tool_call>...</tool_call> 모두 포함
        # <|BOT|> = 125039, <|EOT|> = 125040
        bot_token_id = 125039
        eot_token_id = 125040

        assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
        assistant_len = len(assistant_tokens)

        i = 0
        while i < len(input_ids):
            if input_ids[i] == bot_token_id:
                # <|BOT|> 다음 토큰들이 "assistant\n"인지 확인
                if i + 1 + assistant_len <= len(input_ids):
                    next_tokens = input_ids[i + 1: i + 1 + assistant_len]
                    if next_tokens == assistant_tokens:
                        # assistant 응답 시작: <|BOT|>assistant\n 이후부터
                        start = i + 1 + assistant_len
                        # <|EOT|>까지 labels 설정
                        # <think>, <tool_call> 등 모든 내용이 학습 대상
                        j = start
                        while j < len(input_ids):
                            labels[j] = input_ids[j]
                            if input_ids[j] == eot_token_id:
                                break
                            j += 1
                        i = j + 1
                        continue
            i += 1

        tokenized["labels"] = labels
        return tokenized

    combined = combined.map(
        tokenize_chat,
        remove_columns=["messages"],
        num_proc=1,
        desc="Tokenizing SFT data",
    )

    # 빈 샘플 제거 (labels가 전부 -100인 경우)
    combined = combined.filter(
        lambda x: any(l != -100 for l in x["labels"]),
        num_proc=1,
        desc="Filtering empty labels",
    )

    print_rank0(f"  Final tokenized samples: {len(combined):,}")

    return combined


def main():
    """SFT 메인 함수."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./kogum-0.5B-16k-sft")
    parser.add_argument("--tokenizer_name", type=str, default="KORMo-Team/KORMo-tokenizer")
    parser.add_argument("--run_name", type=str, default="kogum-0.5B-16k-sft")
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=7e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=5)
    # 샘플링 설정 (한/영 7:3 비율, 총 ~400K)
    parser.add_argument("--nemo_ko_samples", type=int, default=200000)
    parser.add_argument("--if_bilingual_samples", type=int, default=80000)
    parser.add_argument("--nemotron_chat_samples", type=int, default=30000)
    parser.add_argument("--nemotron_code_samples", type=int, default=20000)
    parser.add_argument("--nemotron_math_samples", type=int, default=20000)
    parser.add_argument("--nemotron_tool_samples", type=int, default=15000)
    parser.add_argument("--smoltalk_samples", type=int, default=35000)
    args = parser.parse_args()

    set_seed(args.seed)

    print_rank0("=" * 80)
    print_rank0("KoGum 0.5B SFT (Supervised Fine-Tuning)")
    print_rank0("=" * 80)

    # 1. Tokenizer 로드
    print_rank0(f"\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.padding_side = "right"
    print_rank0(f"  Vocab size: {tokenizer.vocab_size:,}")
    print_rank0(f"  PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

    # 2. Model 로드
    print_rank0(f"\n[2/5] Loading model...")
    print_rank0(f"  Checkpoint: {args.model_name_or_path}")
    model = KoGumForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    num_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"  Total parameters: {num_params:,} ({num_params / 1e9:.2f}B)")

    # 3. Dataset 로드 + Tokenize
    print_rank0(f"\n[3/5] Loading and tokenizing SFT datasets...")
    train_dataset = load_sft_datasets(
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        nemo_ko_samples=args.nemo_ko_samples,
        if_bilingual_samples=args.if_bilingual_samples,
        nemotron_chat_samples=args.nemotron_chat_samples,
        nemotron_code_samples=args.nemotron_code_samples,
        nemotron_math_samples=args.nemotron_math_samples,
        nemotron_tool_samples=args.nemotron_tool_samples,
        smoltalk_samples=args.smoltalk_samples,
    )

    # 4. Training Arguments
    print_rank0(f"\n[4/5] Setting up training arguments...")
    training_args = KoGumTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        run_name=args.run_name,

        num_train_epochs=args.num_train_epochs,
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
        gradient_checkpointing_kwargs={"use_reentrant": True},

        max_grad_norm=args.max_grad_norm,

        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        group_by_length=True,
    )

    print_rank0(f"  Epochs: {training_args.num_train_epochs}")
    print_rank0(f"  Learning rate: {training_args.learning_rate}")
    print_rank0(f"  Save total limit: {args.save_total_limit}")

    # 5. Trainer
    print_rank0(f"\n[5/5] Initializing trainer...")
    trainer = KoGumTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
    )

    print_rank0(f"  Trainer initialized")

    # 학습 시작
    print_rank0("\n" + "=" * 80)
    print_rank0("Starting SFT...")
    print_rank0(f"  Base model: {args.model_name_or_path}")
    print_rank0(f"  Epochs: {args.num_train_epochs}")
    print_rank0(f"  Learning rate: {args.learning_rate}")
    print_rank0("=" * 80)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 최종 모델 저장
    print_rank0("\n" + "=" * 80)
    print_rank0("Saving final model...")
    print_rank0("=" * 80)

    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print_rank0(f"  Model saved to: {final_model_dir}")
    print_rank0("\nSFT completed!")


if __name__ == "__main__":
    main()
