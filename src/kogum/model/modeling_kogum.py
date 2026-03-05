"""KoGum model implementation.

Korean-centric decoder-only transformer (LLaMA-style architecture).

구성 요소:
    KoGumRMSNorm         - Root Mean Square Layer Normalization (Pre-LN)
    KoGumRotaryEmbedding - Rotary Position Embeddings (RoPE)
    KoGumAttention       - Grouped Query Attention (GQA) + FlashAttention (SDPA)
    KoGumMLP             - SwiGLU Feed-Forward Network
    KoGumDecoderLayer    - Single transformer decoder layer (RMSNorm → Attn → +residual → RMSNorm → MLP → +residual)
    KoGumModel           - Embedding + 24 DecoderLayers + Final RMSNorm
    KoGumForCausalLM     - KoGumModel + LM Head (untied) + CrossEntropyLoss + Token Accuracy

0.5B 기본 설정 (kogum_0.5B_16k_kormo.yaml):
    vocab=125041, hidden=1024, layers=24, heads=16Q/8KV, intermediate=5120
    context=16384, rope_theta=500000, bf16, gradient_checkpointing
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,            
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache

from .configuration_kogum import KoGumConfig


class KoGumRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    LayerNorm과 달리 평균(mean)을 빼지 않고, 제곱 평균의 역수근(rsqrt)만 곱함.
    연산이 더 적어서 빠르고, LLM에서 성능 차이 없음 (LLaMA, Gemma 등에서 사용).

    수식: RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class KoGumRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Query/Key 벡터를 위치(position)에 따라 회전시키는 상대적 위치 인코딩.
    학습 파라미터 없이 위치 정보를 인코딩함 (LLaMA, GPT-NeoX 등에서 사용).

    동작 원리:
        1. 각 position에 대해 주파수 계산: freq_i = 1 / (theta ^ (2i/dim))
        2. cos/sin 값을 미리 캐싱 (max_position_embeddings까지)
        3. forward에서 Q, K에 회전 적용: q_rot = q * cos + rotate_half(q) * sin

    Args:
        dim: head_dim (64)
        max_position_embeddings: 최대 위치 (16384)
        base: theta 값 (500000.0, 클수록 긴 context 지원)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device] = None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)

        cos = self.cos_cached[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads (for GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class KoGumAttention(nn.Module):
    """Grouped Query Attention (GQA) with FlashAttention (SDPA).

    GQA: Query heads(16)를 KV heads(8)보다 많이 사용하여 KV 캐시 메모리 절약.
    각 KV head가 2개의 Q head를 담당 (num_kv_groups = 16/8 = 2).

    FlashAttention: BxHxLxL attention matrix를 메모리에 올리지 않고 tiled 계산.
    16K context에서 Math backend는 int32 오버플로우 발생하므로 반드시 Flash 사용.

    Forward 흐름:
        hidden → Q/K/V projection → RoPE 적용 → KV repeat (8→16) → SDPA → O projection
    """

    def __init__(self, config: KoGumConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = KoGumRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # Repeat K/V for GQA
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        # Force contiguous for stability (prevents cuBLAS invalid argument errors)
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Get actual KV sequence length (after cache update)
        kv_seq_len = key_states.shape[-2]

        # Adjust attention mask to match KV sequence length
        # attention_mask from model is [batch, 1, seq_len, seq_len] for training
        # But we need [batch, 1, seq_len, kv_seq_len] when using KV cache
        if attention_mask is not None and attention_mask.shape[-1] != kv_seq_len:
            # During inference with KV cache: extend mask
            if kv_seq_len > seq_len:
                # Pad mask on the left to match kv_seq_len
                pad_size = kv_seq_len - attention_mask.shape[-1]
                attention_mask = F.pad(attention_mask, (pad_size, 0), value=0.0)
            # During training: mask should already match
            # If mismatch occurs during gradient checkpointing, create new mask
            elif kv_seq_len < attention_mask.shape[-1]:
                # Truncate mask (shouldn't happen in normal training)
                attention_mask = attention_mask[..., :, :kv_seq_len]

        # Use SDPA with FlashAttention for 16K context
        # FlashAttention NEVER materializes the B×H×L×L attention weights matrix
        # This avoids int32 indexing limit issues (2^31-1 elements)
        #
        # Math backend DOES materialize attention weights for softmax, which fails for 16K:
        #   B=2, H=22, L=16384 → 2×22×16384×16384 = 11.8B elements > int32 limit
        #
        # FlashAttention requirements:
        # - Query/Key/Value must be contiguous (already enforced above)
        # - Mask must be None (use is_causal=True) OR bool tensor
        # - Gradient checkpointing is supported

        # Check if we can use is_causal shortcut (no padding mask)
        use_causal_shortcut = (attention_mask is None)

        # Use FlashAttention with bfloat16
        # FlashAttention requirements:
        # - Query/Key/Value must be bfloat16 or float16 (NOT float32)
        # - Sequence length should be multiple of 8 (for optimal performance)
        # - For long context (16K+), only FlashAttention avoids int32 overflow
        #
        # Context size check for Math backend: B * H * L * L elements
        # - 8K: 2 * 22 * 8192 * 8192 = 2.95B elements (< 2^31, Math safe but slower)
        # - 16K: 2 * 22 * 16384 * 16384 = 11.8B elements (> 2^31, Math OVERFLOW!)
        #
        # Since model is in bfloat16, FlashAttention will be used automatically

        seq_length = query_states.shape[2]

        # For 16K+, disable Math to avoid int32 overflow
        # For 8K, allow Math as fallback (but Flash will be preferred with bfloat16)
        enable_math_backend = (seq_length <= 8192)

        # Ensure attention_mask dtype matches query dtype (prevents dtype mismatch in SDPA)
        if attention_mask is not None and attention_mask.dtype != query_states.dtype:
            attention_mask = attention_mask.to(query_states.dtype)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,              # Primary: FlashAttention (fastest, most memory efficient)
            enable_math=enable_math_backend,  # Fallback for 8K only (slower but safe)
            enable_mem_efficient=False      # Disable (not as fast as Flash)
        ):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=use_causal_shortcut,
            )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class KoGumMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    일반 FFN(Linear→ReLU→Linear)과 달리 gate 메커니즘으로 성능 향상.
    파라미터는 3개 Linear(gate, up, down)로 일반 FFN 대비 ~50% 많지만,
    같은 파라미터 수 기준 성능이 더 좋아서 LLaMA 계열 표준.

    수식: output = down(SiLU(gate(x)) * up(x))
    차원: 1024 → 5120(gate, up) → 5120(element-wise mul) → 1024(down)
    """

    def __init__(self, config: KoGumConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class KoGumDecoderLayer(nn.Module):
    """Single transformer decoder layer (Pre-LN architecture).

    흐름: x → RMSNorm → Attention → +residual → RMSNorm → SwiGLU MLP → +residual
    Pre-LN: Norm을 sublayer 앞에 배치 (Post-LN보다 학습 안정성 높음).
    """

    def __init__(self, config: KoGumConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KoGumAttention(config, layer_idx)
        self.mlp = KoGumMLP(config)
        self.input_layernorm = KoGumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KoGumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # Self Attention with Pre-LN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with Pre-LN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class KoGumPreTrainedModel(PreTrainedModel):
    """KoGum 모델의 베이스 클래스.

    HuggingFace PreTrainedModel을 상속하여 weight 초기화(_init_weights),
    gradient checkpointing, from_pretrained/save_pretrained 등 기능 제공.
    """

    config_class = KoGumConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["KoGumDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KoGumModel(KoGumPreTrainedModel):
    """KoGum transformer 본체 (decoder-only).

    구조: Embedding(125041→1024) → 24× DecoderLayer → Final RMSNorm
    LM Head는 포함하지 않음 (KoGumForCausalLM에서 추가).
    """

    def __init__(self, config: KoGumConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [KoGumDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = KoGumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def _create_intra_document_causal_mask(
        self,
        document_ids: torch.LongTensor,
        seq_length: int,
        past_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create intra-document causal mask.

        Tokens can only attend to previous tokens in the SAME document.

        Args:
            document_ids: [batch_size, seq_length] - document ID for each token

        Returns:
            4D mask: [batch_size, 1, seq_length, seq_length]
        """
        batch_size = document_ids.shape[0]

        # Create causal mask (lower triangular)
        causal_mask = torch.tril(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device)
        )  # [seq_length, seq_length]

        # Create document mask (same document)
        # document_ids: [batch_size, seq_length]
        # Expand to [batch_size, seq_length, 1] and [batch_size, 1, seq_length]
        # Compare: [batch_size, seq_length, seq_length]
        doc_mask = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)  # [batch, seq, seq]

        # Combine: can attend if (1) causal AND (2) same document
        combined_mask = causal_mask.unsqueeze(0) & doc_mask  # [batch, seq, seq]

        # Convert to additive mask
        mask = combined_mask.to(dtype)
        mask = mask.masked_fill(~combined_mask, float('-inf'))
        mask = mask.masked_fill(combined_mask, 0.0)

        # Add head dimension: [batch_size, 1, seq_length, seq_length]
        return mask.unsqueeze(1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        document_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CRITICAL: Disable KV cache during training to avoid gradient checkpointing bugs
        # KV cache should only be used during inference
        if self.training:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape # (batch_size, seq_length)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        # Handle past key values (only during inference when use_cache=True)
        # 학습 시: use_cache = False -> past_key_values = None, past_length = 0
        # 추론 시: 이전에 생성한 토큰 수를 past_length로 가져옴. 예를 들어 이미 100 토큰을 생성했으면 past_length = 100
        past_length = 0
        if use_cache and past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.get_seq_length()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_length = past_key_values.get_seq_length()
        elif use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        else:
            # Training mode: no KV cache
            past_key_values = None

        # Position IDs
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) # |position_ids|: [batch_size, seq_length]

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds # |hidden_states| = [batch_size, seq_length, hidden_size]

        # Prepare attention mask
        # For intra-document masking: use document_ids to prevent cross-document attention
        # For standard causal: DO NOT create mask here (handled in attention layer directly to save memory)
        causal_mask = None
        if document_ids is not None:                                                # Case 1: intra-document masking
            # Intra-document causal masking
            causal_mask = self._create_intra_document_causal_mask(
                document_ids, seq_length, past_length, hidden_states.dtype, hidden_states.device
            )
        elif attention_mask is not None:                                            # Case 2: padding이 있는 경우
            # Standard attention mask (for padding, etc.)
            causal_mask = self._prepare_attention_mask(attention_mask, seq_length, past_length)
        # else: No mask - causal masking will be done in attention layer directly   # Case 3: packed sequences

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 중간 activation을 저장하지 않고, backward 시 다시 계산
                # 메모리 절약 (GPU 80GB에서 16K context 가능하게 해줌)
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__, # 함수 객체
                    hidden_states,          # 이하 함수에 전달할 인자들
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_length: int,
        past_length: int,
    ) -> torch.Tensor:
        """Prepare 4D causal attention mask for SDPA.

        Returns float mask with -inf for positions to mask out.
        Uses bf16 dtype to match model precision for stability.
        """
        # past_length: 이미 처리된 과거 토큰 (kv cache에 저장된 길이)
        # seq_length: 이번 forward에 처리할 현재 토큰 길이
        # total_length: past + current, KV 시퀀스 전체 길이 (캐시 업데이트 후)
        # 우리가 만드는 마스크 shape: [batch_size, 1, seq_length, total_length]
        batch_size = attention_mask.shape[0]
        total_length = seq_length + past_length

        # Create causal mask (True = cannot attend to future)
        # false: attend, true: mask out
        # triu: 위쪽 삼각형만 남기고 나머지는 False로 한다
        # diagonal = k는 대각선을 k칸 위로 올린 선부터 위쪽을 살린다.
        causal_mask = torch.triu(
            torch.ones((seq_length, total_length), dtype=torch.bool, device=attention_mask.device),
            diagonal=past_length + 1 # 자기 자신까지는 허용, 그 다음부터는 미래라서 차단(True)
        ) # |causal_mask| = [seq_length, total_length]

        # Expand attention mask to 4D (True = cannot attend to padding)
        expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, total_length)
        expanded_mask = expanded_mask == 0

        # Combine causal and padding masks (True = cannot attend)
        combined_mask = causal_mask[None, None, :, :] | expanded_mask

        # Convert to float mask using bf16 (matches model dtype for stability)
        # Don't use float32 with -inf as it can cause precision issues in backward
        mask = torch.zeros((batch_size, 1, seq_length, total_length),
                          dtype=torch.bfloat16,
                          device=attention_mask.device)
        mask.masked_fill_(combined_mask, float("-inf"))

        return mask


class KoGumForCausalLM(KoGumPreTrainedModel, GenerationMixin):
    """KoGum + LM Head (next-token prediction).

    구조: KoGumModel + Linear(1024→125041, untied from embedding)

    Training 시:
        - labels가 주어지면 CrossEntropyLoss 계산 (shifted: logits[:-1] vs labels[1:])
        - token accuracy를 forward 내에서 계산하여 _token_accuracy에 저장
        - logits는 반환하지 않음 (125K vocab × 16K seq = ~7.6GB 메모리 절약)

    Inference 시:
        - logits 반환, KV cache 지원 (GenerationMixin으로 generate() 사용 가능)
    """

    def __init__(self, config: KoGumConfig):
        super().__init__(config)
        self.model = KoGumModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def get_decoder(self) -> KoGumModel:
        return self.model

    def set_decoder(self, decoder: KoGumModel):
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        document_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] # |hidden_states| = [batch_size, seq_length, hidden_size] = [2, 16384, 1024]

        loss = None
        logits = None
        if labels is not None:
            logits = self.lm_head(hidden_states) # |logits| = [batch_size, seq_length, vocab_size] = [2, 16384, 125041] 

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute token accuracy BEFORE flattening (no extra memory)
            with torch.no_grad():
                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100 # mask: True (유효 토큰), mask: False (무시할 토큰)
                correct = (predictions == shift_labels) & mask
                total = mask.sum()
                self._token_accuracy = (correct.sum().float() / total).item() if total > 0 else 0.0

            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

            # During training: do NOT return logits (125K vocab × 16K seq = ~7.6GB!)
            if self.training:
                if not return_dict:
                    return (loss,)
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=None,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                )
        else:
            # Inference: compute logits
            logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            # Only use last token if we have cache
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

        return model_inputs
