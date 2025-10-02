import os
import time
import timeit

from pathlib import Path

import torch

from transformers import AutoConfig, AutoTokenizer
from lxr_modeling_llama_fourier import LlamaForCausalLM
from lxr_cache_utils_fourier import DynamicCache


def print_gpu_memory_info():
    # 获取当前设备的显存信息
    current_device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    allocated_memory = torch.cuda.max_memory_allocated(current_device)
    
    # 转换为GB单位
    total_memory_gb = total_memory / (1024 ** 3)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    
    # 计算使用百分比
    usage_percentage = (allocated_memory / total_memory) * 100
    
    print(f"Total Mem: {total_memory_gb:.2f} GB, Allocated Mem: {allocated_memory_gb:.2f} GB ({usage_percentage:.2f}% used)")


class ExtraConfig:
    def __init__(self, max_new_tokens, maxlocallen, maxmidstates, numinittokens, non_critical_dims_path, 
                 max_position_embeddings, num_key_value_heads):
        self.max_new_tokens = max_new_tokens
        self.maxlocallen = maxlocallen
        self.maxmidstates = maxmidstates
        self.numinittokens = numinittokens
        self.non_critical_dims_path = non_critical_dims_path
        self.max_position_embeddings = max_position_embeddings
        self.num_key_value_heads = num_key_value_heads  #Newly added for HippoAttention


def model_gen(model, config, input_ids, max_new_tokens, ):

    generated_sequence = input_ids
    init_len = input_ids.size(1)

    extra_config = ExtraConfig(max_new_tokens=max_new_tokens, numinittokens=4, maxlocallen=1020, 
                               maxmidstates=1024,  # 512 * 2
                               non_critical_dims_path="/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_402/hippo_fourier/dimdifferjson_32k/non_critical_dims_hippofourier_kvdiffer6_512mid_splithead.json",
                               max_position_embeddings=config.max_position_embeddings, num_key_value_heads=config.num_key_value_heads, 
                               )
    past_key_values = DynamicCache(extra_config)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_sequence = torch.cat([generated_sequence, input_ids], dim=-1)
            past_key_values = outputs.past_key_values
                    
            if input_ids.item() == eos_token_id:
                break
    
    return generated_sequence[:, init_len:]


if __name__ == "__main__":

    device = torch.device("cuda:0")
    model_name = "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/"
    config = AutoConfig.from_pretrained(model_name)

    model = LlamaForCausalLM.from_pretrained(model_name, config=config, device_map=device, 
                                             torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    begin = "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards."
    end = "What is the special magic number for wandering-age mentioned in the provided text? The special magic number for wandering-age mentioned in the provided text is"
    needle = "One of the special magic numbers for wandering-age is: 8090293."
    hazy = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

    text_repeat_num = 640  # [160, 320, 640, 1280, ]  # 4k, 8k, 16k, 32k, 

    text = '\n'.join([hazy] * (text_repeat_num // 2))
    text = '\n'.join([begin, text, needle, text, end])

    max_new_tokens = 10

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)  # 转移到GPU

    input_ids = inputs["input_ids"].to(model.device)

    generated_sequence = model_gen(model, config, input_ids, max_new_tokens, )

    generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)

    print(f"N_CTX: {input_ids.size(1)}, FlashFourierAttention v2, Generated text: {generated_text}")

    repeats = 10

    prefill_time = 0
    for i in range(repeats):
        start_time = time.time()
        model_gen(model, config, input_ids, max_new_tokens=1)
        torch.cuda.synchronize()
        end_time = time.time()
        prefill_time += (end_time - start_time)
    prefill_time /= repeats

    prefill_decode10_time = 0
    for i in range(repeats):
        start_time = time.time()
        model_gen(model, config, input_ids, max_new_tokens=11)
        torch.cuda.synchronize()
        end_time = time.time()
        prefill_decode10_time += (end_time - start_time)
    prefill_decode10_time /= repeats

    decode_speed = (prefill_decode10_time - prefill_time) / 10

    print(f"Prefill: {prefill_time:.3f} s, Prefill + Decode 10 = {prefill_decode10_time:.3f} ms, Decode time: {decode_speed:.3f} ms/token")
    print_gpu_memory_info()
    print('\n')

