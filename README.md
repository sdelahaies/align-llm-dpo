Align LLM with direct preference optimization (DPO)

requirements

```
git clone https://github.com/sdelahaies/align-llm-dpo.git
cd align-llm-dpo
uv venv python 3.10.12
source .venv/bin/activate
```

```
uv pip install "torch==2.4.1" \
    tensorboard \
    flash-attn \
    "liger-kernel==0.4.2" \
    "setuptools<71.0.0" \
    "deepspeed==0.15.4" \
    openai \
    "lm-eval[api]==0.4.5"
    python-dotenv
```

```
uv pip install  --upgrade \
  "transformers==4.46.3" \
  "datasets==3.1.0" \
  "accelerate==1.1.1" \
  "bitsandbytes==0.44.1" \
  "trl==0.12.1" \
  "peft==0.13.2" \
  "lighteval==0.6.2" \
  "hf-transfer==0.1.8"
```