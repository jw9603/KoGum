"""KoGum Trainer implementation.

HuggingFace Trainer를 확장하여 KoGum 특화 기능 추가:
- Weight decay 선택적 적용 (LayerNorm, Embedding 제외)
- Token-level accuracy 계산
- 8-bit optimization 지원
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import Trainer, logging
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from kogum.model.modeling_kogum import KoGumRMSNorm


logger = logging.get_logger(__name__)


class KoGumTrainer(Trainer):
    """KoGum 학습을 위한 커스텀 Trainer.

    HuggingFace Trainer를 상속받아 다음 기능 추가:
    1. Selective weight decay (LayerNorm/Embedding 제외)
    2. Token accuracy 계산
    3. 8-bit AdamW 지원 (메모리 절약)

    사용 예시:
        >>> from kogum.train import KoGumTrainer, KoGumTrainingArguments
        >>> args = KoGumTrainingArguments(output_dir="./output")
        >>> trainer = KoGumTrainer(
        ...     model=model,
        ...     args=args,
        ...     train_dataset=train_dataset,
        ... )
        >>> trainer.train()
    """

    def __init__(self, spike_detector=None, **kwargs):
        super().__init__(**kwargs)
        # 메트릭 추적용 딕셔너리
        # train/eval 모드별로 각 스텝의 메트릭을 누적 후 평균 계산
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Spike detector (배치 저장용)
        self.spike_detector = spike_detector

    # =========================================================================
    # Weight Decay 설정
    # =========================================================================
    def get_decay_parameter_names(self, model) -> List[str]:
        """Weight decay를 적용할 파라미터 이름 목록을 반환합니다.

        Weight decay는 일반적으로 Linear layer의 weight에만 적용하고,
        다음 파라미터에는 적용하지 않습니다:
        - LayerNorm / RMSNorm의 weight
        - 모든 bias
        - Embedding

        이유:
        - LayerNorm은 이미 정규화 역할을 하므로 weight decay 불필요
        - Bias는 작은 값이라 weight decay 효과 미미
        - Embedding은 sparse하므로 decay하면 성능 저하

        Args:
            model: 학습할 모델

        Returns:
            Weight decay를 적용할 파라미터 이름 리스트
        """
        # Weight decay를 적용하지 않을 모듈 타입
        # ALL_LAYERNORM_LAYERS: transformers에서 제공하는 모든 LayerNorm 타입
        # + Embedding, RMSNorm
        forbidden_layer_types = tuple(ALL_LAYERNORM_LAYERS) + (
            nn.Embedding,
            KoGumRMSNorm,
        )

        # get_parameter_names: forbidden 타입을 제외한 모든 파라미터 이름 반환
        decay_parameters = get_parameter_names(
            model,
            forbidden_layer_types=forbidden_layer_types,
        )

        # bias 파라미터는 수동으로 제외
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        logger.debug(f"Parameters with weight decay: {len(decay_parameters)}")
        return decay_parameters

    # =========================================================================
    # Optimizer 생성
    # =========================================================================
    def create_optimizer(self):
        """Optimizer를 생성합니다.

        파라미터를 두 그룹으로 나누어 optimizer에 전달:
        1. Weight decay 적용 그룹 (대부분의 linear weights)
        2. Weight decay 미적용 그룹 (bias, norms, embeddings)

        추가로 bitsandbytes 8-bit optimizer 사용 시,
        Embedding은 32-bit로 유지합니다 (정확도 유지).

        Returns:
            생성된 optimizer
        """
        opt_model = self.model

        if self.optimizer is None:
            # Step 1: Weight decay 적용 대상 파라미터 목록 가져오기
            decay_parameters = self.get_decay_parameter_names(opt_model)

            # Step 2: 파라미터를 두 그룹으로 분리
            optimizer_grouped_parameters = [
                {
                    # Group 1: Weight decay 적용
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    # Group 2: Weight decay 미적용
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # Step 3: Optimizer 클래스 및 kwargs 가져오기
            # (TrainingArguments의 optim 설정 기반)
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            # Step 4: Optimizer 생성
            # 중복된 params 인자 제거 (우리가 직접 관리)
            for key in ["params", "model", "optimizer_dict"]:
                if key in optimizer_kwargs:
                    optimizer_kwargs.pop(key)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # =====================================================================
            # bitsandbytes 8-bit optimizer 특별 처리
            # =====================================================================
            # Embedding은 8-bit가 아닌 32-bit(FP32)로 최적화
            # 이유: Embedding은 sparse하고, 8-bit로 하면 정확도 손실 큼
            if "bitsandbytes" in str(optimizer_cls):
                try:
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped_params = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            # 이 Embedding 모듈의 파라미터 개수 계산
                            module_params = sum(
                                {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                            )
                            skipped_params += module_params

                            # 이 모듈의 weight는 32-bit로 최적화하도록 override
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32}
                            )
                            logger.debug(
                                f"bitsandbytes: {module} will be optimized in FP32"
                            )

                    logger.info(
                        f"Skipped 8-bit optimization for embeddings: {skipped_params / 2**20:.2f}M params"
                    )
                except ImportError:
                    logger.warning("bitsandbytes not installed, skipping 8-bit optimization")

        return self.optimizer

    # =========================================================================
    # Loss 계산 + Token Accuracy
    # =========================================================================
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Loss를 계산하고 token-level accuracy도 함께 계산합니다.

        Causal LM의 성능을 평가하기 위해:
        - Loss (cross entropy): 모델이 다음 토큰을 얼마나 잘 예측하는가
        - Token accuracy: 예측한 토큰이 실제 정답과 얼마나 일치하는가

        Args:
            model: 학습 중인 모델
            inputs: 배치 입력 (input_ids, labels 등)
            return_outputs: True면 (loss, outputs) 반환
            num_items_in_batch: 배치 내 아이템 수 (선택)

        Returns:
            loss 또는 (loss, outputs)
        """
        mode = "train" if self.model.training else "eval"

        # Step 1: 기본 loss 계산 (부모 클래스 메서드)
        # return_outputs=True로 강제하여 outputs 받아옴
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Step 2: Token accuracy 계산
        # 모델이 forward에서 _token_accuracy를 계산하여 저장함 (logits 반환 없이 메모리 절약)
        raw_model = model.module if hasattr(model, 'module') else model
        if hasattr(raw_model, '_token_accuracy'):
            self._metrics[mode]["mean_token_accuracy"].append(raw_model._token_accuracy)

        # =====================================================================
        # Spike 감지 시 배치 저장 (train 모드에서만)
        # =====================================================================
        if mode == "train" and self.spike_detector is not None:
            # processing_class를 사용 (tokenizer의 새로운 이름)
            tokenizer = getattr(self, 'processing_class', None) or self.tokenizer

            # Gradient norm 가져오기 (optimizer step 후에만 사용 가능)
            grad_norm = 0.0
            if hasattr(self.state, 'last_grad_norm') and self.state.last_grad_norm is not None:
                grad_norm = self.state.last_grad_norm

            # Convert loss to scalar (handle DataParallel case where loss may have multiple elements)
            if torch.is_tensor(loss):
                loss_scalar = loss.mean().item() if loss.numel() > 1 else loss.item()
            else:
                loss_scalar = loss

            self.spike_detector.check_and_save_batch(
                step=self.state.global_step,
                loss=loss_scalar,
                grad_norm=grad_norm,
                inputs=inputs,
                outputs=outputs,
                tokenizer=tokenizer,
            )

        return (loss, outputs) if return_outputs else loss

    # =========================================================================
    # 로깅
    # =========================================================================
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """메트릭을 로깅합니다.

        수집된 메트릭(token accuracy 등)의 평균을 계산하여 로그에 추가합니다.

        Args:
            logs: 로깅할 메트릭 딕셔너리
            start_time: 시작 시간 (선택)
        """
        mode = "train" if self.model.training else "eval"

        # Step 1: 누적된 메트릭의 평균 계산
        # 예: mean_token_accuracy = [0.65, 0.68, 0.67] → 0.667
        metrics = {
            key: sum(val) / len(val) if len(val) > 0 else 0.0
            for key, val in self._metrics[mode].items()
        }

        # Step 2: logs에 추가
        logs.update(metrics)

        # Step 3: 학습 진행도 추가
        if self.args.include_num_input_tokens_seen:
            # 총 입력 토큰 수 (학습 진행도 파악용)
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        # Epoch는 streaming mode에서 의미 없으므로 로깅하지 않음
        # if self.state.epoch is not None:
        #     logs["epoch"] = round(self.state.epoch, 2)

        # Step 4: 글로벌 스텝 추가
        logs["step"] = self.state.global_step
        output = {**logs}

        # Step 5: 로그 기록 및 콜백 실행
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        # Step 6: 메트릭 초기화 (다음 로깅 주기를 위해)
        self._metrics[mode].clear()
