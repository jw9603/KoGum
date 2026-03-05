#!/usr/bin/env python3
"""학습 중 스파이크를 감지하고 실제 배치 데이터를 저장하는 클래스"""

import json
import os
from pathlib import Path
from typing import Optional

import torch


class BatchSpikeDetector:
    """Loss/Accuracy 스파이크를 감지하고 실제 배치 데이터를 저장

    스트리밍 모드에서도 사용 가능하도록 실시간으로 배치 데이터를 저장합니다.

    Args:
        output_dir: 배치 데이터를 저장할 디렉토리
        loss_threshold: Loss 증가 임계값 (배수). 예: 2.0 = 이동평균 대비 2.0배 증가 시 감지
        grad_norm_threshold: Gradient norm 증가 임계값 (배수). 예: 2.0 = 이동평균 대비 2.0배 증가 시 감지
        window_size: 이동 평균을 계산할 윈도우 크기 (기본: 10 steps)
        consecutive_steps: 연속으로 높은 loss가 유지되어야 하는 step 수 (기본: 2 steps)
        save_batch_samples: 배치에서 저장할 샘플 수 (기본: 5개)
    """

    def __init__(
        self,
        output_dir: str = "./spike_batches",
        loss_threshold: float = 2.0,
        grad_norm_threshold: float = 2.0,
        window_size: int = 10,
        consecutive_steps: int = 2,
        save_batch_samples: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loss_threshold = loss_threshold
        self.grad_norm_threshold = grad_norm_threshold
        self.window_size = window_size
        self.consecutive_steps = consecutive_steps
        self.save_batch_samples = save_batch_samples

        # 메트릭 히스토리
        self.loss_history = []
        self.grad_norm_history = []

        # 연속 high loss 카운터
        self.high_loss_counter = 0

        # 마지막으로 저장한 step (중복 저장 방지)
        self.last_saved_step = -1

        print(f"[BatchSpikeDetector] Initialized")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Loss threshold: {self.loss_threshold}x (moving average)")
        print(f"  Grad norm threshold: {self.grad_norm_threshold}x (moving average)")
        print(f"  Window size: {self.window_size} steps")
        print(f"  Consecutive steps: {self.consecutive_steps} steps")
        print(f"  Batch samples to save: {self.save_batch_samples}")

    def check_and_save_batch(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        inputs: dict,
        outputs,
        tokenizer,
    ):
        """Loss/Grad norm 스파이크를 감지하고 배치를 저장

        Args:
            step: 현재 global step
            loss: 현재 loss 값
            grad_norm: 현재 gradient norm 값
            inputs: 배치 입력 (input_ids, labels, attention_mask 등)
            outputs: 모델 출력
            tokenizer: 토크나이저 (텍스트 디코딩용)
        """

        # 중복 저장 방지
        if step == self.last_saved_step:
            return

        # 히스토리에 추가
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

        # 윈도우 크기 유지
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        if len(self.grad_norm_history) > self.window_size:
            self.grad_norm_history.pop(0)

        # 최소 window_size 개 이상의 데이터가 있어야 스파이크 감지 가능
        if len(self.loss_history) < self.window_size:
            return

        # 이동 평균 계산 (현재 step 제외)
        avg_loss = sum(self.loss_history[:-1]) / len(self.loss_history[:-1])
        avg_grad_norm = sum(self.grad_norm_history[:-1]) / len(self.grad_norm_history[:-1])

        # 스파이크 조건: loss가 높음 OR grad_norm이 급증
        is_high_loss = loss > avg_loss * self.loss_threshold
        is_high_grad = grad_norm > avg_grad_norm * self.grad_norm_threshold

        if is_high_loss or is_high_grad:
            self.high_loss_counter += 1
        else:
            self.high_loss_counter = 0  # 리셋

        # 연속 N step 동안 높으면 진짜 spike
        if self.high_loss_counter >= self.consecutive_steps:
            self._save_batch_data(
                step=step,
                loss=loss,
                avg_loss=avg_loss,
                grad_norm=grad_norm,
                avg_grad_norm=avg_grad_norm,
                inputs=inputs,
                outputs=outputs,
                tokenizer=tokenizer,
            )
            self.last_saved_step = step
            self.high_loss_counter = 0  # 저장 후 리셋

    def _save_batch_data(
        self,
        step: int,
        loss: float,
        avg_loss: float,
        grad_norm: float,
        avg_grad_norm: float,
        inputs: dict,
        outputs,
        tokenizer,
    ):
        """스파이크가 발생한 배치의 실제 데이터 저장"""

        # rank 0에서만 저장 (분산 학습 시)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        print("\n" + "=" * 80)
        print(f"⚠️  SPIKE DETECTED at step {step}!")
        print("=" * 80)
        print(f"  Current loss: {loss:.4f}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Loss ratio: {loss / avg_loss:.2f}x")
        print(f"  Current grad_norm: {grad_norm:.4f}")
        print(f"  Average grad_norm: {avg_grad_norm:.4f}")
        print(f"  Grad norm ratio: {grad_norm / avg_grad_norm:.2f}x")
        print(f"  Consecutive high steps: {self.consecutive_steps}")
        print(f"  Saving batch data...")

        # 배치 데이터 추출
        input_ids = inputs.get("input_ids", None)
        labels = inputs.get("labels", None)
        attention_mask = inputs.get("attention_mask", None)

        if input_ids is None:
            print("  ⚠️  Warning: No input_ids in batch")
            return

        # CPU로 이동
        input_ids = input_ids.cpu()
        if labels is not None:
            labels = labels.cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()

        batch_size = input_ids.shape[0]
        num_samples = min(self.save_batch_samples, batch_size)

        # 샘플 데이터 저장
        samples = []
        for i in range(num_samples):
            sample_input_ids = input_ids[i].tolist()
            sample_labels = labels[i].tolist() if labels is not None else None

            # 텍스트 디코딩
            text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)

            # Labels도 디코딩 (pad 토큰 제외)
            if sample_labels is not None:
                valid_labels = [l for l in sample_labels if l != -100]
                label_text = tokenizer.decode(valid_labels, skip_special_tokens=False) if valid_labels else ""
            else:
                label_text = ""

            samples.append({
                "sample_idx": i,
                "input_ids": sample_input_ids,
                "labels": sample_labels,
                "text": text,
                "text_preview": text[:500],
                "label_text_preview": label_text[:500] if label_text else "",
                "length": len(sample_input_ids),
            })

        # 스파이크 정보
        spike_info = {
            "step": step,
            "loss": loss,
            "avg_loss": avg_loss,
            "loss_ratio": loss / avg_loss,
            "grad_norm": grad_norm,
            "avg_grad_norm": avg_grad_norm,
            "grad_norm_ratio": grad_norm / avg_grad_norm,
            "consecutive_steps": self.consecutive_steps,
            "batch_size": batch_size,
            "num_samples_saved": num_samples,
            "samples": samples,
        }

        # JSON 저장
        spike_file = self.output_dir / f"spike_step_{step}.json"
        with open(spike_file, "w", encoding="utf-8") as f:
            json.dump(spike_info, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved {num_samples} samples to: {spike_file}")
        print("=" * 80 + "\n")
