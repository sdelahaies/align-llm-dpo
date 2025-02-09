Align LLM with direct preference optimization (DPO)

requirements

```
git clone https://github.com/sdelahaies/align-llm-dpo.git
cd align-llm-dpo
uv venv python 3.10.12
source .venv/bin/activate
```

## install dependencies

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

## finetune model

```
python run_sft.py --config ft-llama-3.2-3b-qlora.yaml 
```

# merge and push

```
python merge_adapter_weights.py --peft_model_id runs/meta-llama/Llama-3.2-3B-math-orca-qlora-10k-ep1 --push_to_hub True --repository_id Llama-3.2-3B-math-orca-qlora-10k-ep1
```


# run text generation inference server

```
docker run --name tgi --gpus all --shm-size 1g -p 8080:80 -v $volume:/data     --env HF_TOKEN=$(cat ~/.cache/huggingface/token) ghcr.io/huggingface/text-generation-inference:3.1.0 --model-id $model
```

# run evaluation harness 

```
lm_eval --model local-chat-completions   --tasks gsm8k_cot   --model_args model=sylvain471/Llama-3.2-3B-math-orca-qlora-10k-ep1-merged,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=8,max_retries=3,tokenized_requests=False   --apply_chat_template   --fewshot_as_multiturn
```

```
local-chat-completions (model=sylvain471/llama-3.1-8b-math-orca-qlora-10k-ep1,base_url=http://localhost:8080/v1/chat/completions,num_concurrent=8,max_retries=10,tokenized_requests=False), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_cot|      3|flexible-extract|     8|exact_match|↑  |0.5330|±  |0.0137|
|         |       |strict-match    |     8|exact_match|↑  |0.1274|±  |0.0092|
```

# Preference dataset

```
python create_preference_dataset.py --dataset_id philschmid/DMath --sample_size 5000 --generation_model_name_or_path sylvain471/llama-3.1-8b-math-orca-qlora-10k-ep1 --num_solutions 4 --batch_size 16 --token $(cat ~/.cache/huggingface/token)
```