# KoGum Tokenizer Configuration

## Vocabulary
vocab_size: 80038

## Special Token IDs (for model config)
pad_token_id: 0
bos_token_id: 1
eos_token_id: 2
unk_token_id: 3

## Update your model config (kogum_0.5B.yaml)

```yaml
vocab_size: 80038
pad_token_id: 0
bos_token_id: 1
eos_token_id: 2
```

## Python code to update config

```python
from kogum import KoGumConfig

config = KoGumConfig(
    vocab_size=80038,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    # ... other params
)
```
