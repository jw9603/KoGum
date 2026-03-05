"""KoGum model configuration."""

from transformers import PretrainedConfig


class KoGumConfig(PretrainedConfig):
    """Configuration class for KoGum model.

    KoGum is a Korean-centric decoder-only transformer model with:
    - RMSNorm (Pre-LN architecture)
    - Rotary Position Embeddings (RoPE)
    - SwiGLU FFN
    - Grouped Query Attention (GQA)
    - Untied embeddings

    Tokenizer: KORMo 125K BPE vocabulary (70% Korean, 30% English)
    Special tokens: <|PAD|>=125032, <|BOS|>=125030, <|EOT|>(EOS)=125040
    """

    model_type = "kogum"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 125041,
        hidden_size: int = 1024,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 16384,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling: dict = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        bos_token_id: int = 125030,  # <|BOS|>
        eos_token_id: int = 125040,  # <|EOT|>
        pad_token_id: int = 125032,  # <|PAD|>
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # Validate GQA configuration
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_key_value_groups(self) -> int:
        """Number of query heads per key-value head."""
        return self.num_attention_heads // self.num_key_value_heads
