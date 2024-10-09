# LLaMA model with KIVI
import torch
import os
from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM

model_path = '~/models/llama2-7b'
model_path = os.path.expanduser(model_path)

config = LlamaConfig.from_pretrained(model_path)
USE_KIVI = True

if USE_KIVI:
    config.k_bits = 2 # current support 2/4 bit for KV Cache
    config.v_bits = 2 # current support 2/4 bit for KV Cache
    config.group_size = 64
    config.residual_length = 128 # the number of recent fp16 tokens
    CACHE_DIR = './cache'

    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

# Inference
# e.g., model.generate(...)

text = "Write a long long story about a robot learning to love:"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
output_ids = model.generate(input_ids, max_new_tokens=1024)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

print(f"Memory used: {torch.cuda.max_memory_allocated() / 1024**2}")