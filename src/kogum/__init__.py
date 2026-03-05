"""KoGum: Korean-centric Language Model.

A 0.5B parameter decoder-only transformer designed for Korean language.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .model import (
    KoGumConfig,
    KoGumPreTrainedModel,
    KoGumModel,
    KoGumForCausalLM,
)

__version__ = "0.1.0"

# Register with HuggingFace Auto classes
AutoConfig.register("kogum", KoGumConfig)
AutoModel.register(KoGumConfig, KoGumModel)
AutoModelForCausalLM.register(KoGumConfig, KoGumForCausalLM)

__all__ = [
    "KoGumConfig",
    "KoGumPreTrainedModel",
    "KoGumModel",
    "KoGumForCausalLM",
]
