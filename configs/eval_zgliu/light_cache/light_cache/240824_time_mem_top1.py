import argparse
import os
import time
import torch
import numpy
from transformers import AutoTokenizer, AutoConfig
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import datasets

from cache_utils import LightCacheConfig
from light_cache_llama3_1 import LlamaForCausalLM


def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate Llama model with cached flash attention")
    parser.add_argument("--model_path", type=str, default="/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B", help="Path to the model")
    parser.add_argument("--dataset_fp", type=str, default="/remote-home1/rxli/light_cache/data/pg19.json", help="Path to the dataset file")
    parser.add_argument("--save_file", type=str, default='', help="save_file")
    parser.add_argument("--max_out_len", type=int, default=8, help="max_out_len")
    parser.add_argument("--global_size", type=int, default=4, help="global_size")
    parser.add_argument("--local_size", type=int, default=2048, help="local_size")
    parser.add_argument("--mid_size", type=int, default=4, help="mid_size")
    parser.add_argument("--span_size", type=int, default=32, help="span_size")
    parser.add_argument("--chunk_size", type=int, default=4096, help="chunk_size")
    parser.add_argument("--recall_option", type=str, default='default', help="recall_option")
    parser.add_argument("--unique_option", type=str, default='group_unique', help="unique_option")
    parser.add_argument("--recall_clip", type=int, default=64, help="recall_clip")
    parser.add_argument("--input_length", type=int, default=1024000, help="Length of input sequences")
    return parser.parse_args()


class PerformanceMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.max_memory = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.max_memory = torch.cuda.max_memory_allocated() / 1024**2
        self.duration = self.end_time - self.start_time
        self.print_metrics()
        torch.cuda.empty_cache()

    def print_metrics(self):
        print(f"Time taken: {self.duration:.6f} seconds")
        print(f"Memory used: {self.max_memory:.6f} MB")


def main():
    args = parse_args()

    model_path = os.path.expanduser(args.model_path)
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    long_cache_config = {
        'global_size': args.global_size, 'mid_size': args.mid_size, 'span_size': args.span_size, 
        'local_size':args.local_size, 'chunk_size': args.chunk_size, 
        'rope_scaling': None, 'recall_option': args.recall_option, 
        'unique_option': args.unique_option, 'recall_clip': args.recall_clip, }
    long_cache_config = LightCacheConfig(num_key_value_heads=config.num_key_value_heads, 
                                    num_attention_heads=config.num_attention_heads, 
                                    **long_cache_config)
    config.long_cache_config = long_cache_config
    config.pretraining_tp = 1
    model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
                                             config=config, trust_remote_code=True, 
                                             attn_implementation='flash_attention_2')
    model.eval()

    dataset = datasets.load_dataset("json", data_files=args.dataset_fp)
    tokenized_datasets = dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=args.input_length, return_tensors="pt"), num_proc=64)
    tokenized_datasets = tokenized_datasets.filter(lambda x: len(x['input_ids'][0]) >= args.input_length)
    tokenized_datasets = tokenized_datasets['train']

    data_length = len(tokenized_datasets)

    #################### WARM UP ####################
    warmup_datasets = tokenized_datasets.select(range(min(data_length, 2)))
    for example in tqdm(warmup_datasets, desc="Warmup"):
        input_ids = torch.tensor(example['input_ids']).cuda()
        input_ids = input_ids[:, :1024]
        if args.chunk_size > 0:
            with torch.no_grad():
                outputs = model(input_ids)
    
    generation_config = GenerationConfig(
            num_beams=1, do_sample=False, use_cache=True
        )

    # from torch.profiler import profile, ProfilerActivity

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     # with_stack=True, # 可选，将保留完整的调用栈，但相应地文件体积会飞速膨胀
    # ) as p:

    tokenized_datasets = tokenized_datasets.select(range(min(data_length, 1)))
    with PerformanceMetrics() as pm:
        with torch.no_grad():
            time_list = []
            for example in tqdm(tokenized_datasets, desc="Evaluating"):
                input_ids = torch.tensor(example['input_ids']).cuda()
                past_key_values = None
                if args.chunk_size > 0:
                    seq_len = input_ids.shape[-1]
                    mod_len = (seq_len - 1 - args.global_size - args.local_size) % 128
                    local_size = args.local_size - (128 - mod_len)
                    start, end = 0, args.global_size + local_size  # + self.long_cache_config.mid_size 
                    chunk_length = args.chunk_size
                    start_time = time.perf_counter()
                    while start < seq_len - 1:
                        # print(start, end, flush=True)
                        input_chunk = input_ids[:, start:min(end, seq_len - 1)]
                        outputs = model.forward(input_chunk, past_key_values=past_key_values)
                        start, end = end, end + chunk_length
                        past_key_values = outputs.past_key_values
                        # torch.cuda.empty_cache()
                    end_time = time.perf_counter()
                else:
                    start_time = time.perf_counter()
                    outputs = model(input_ids)  # outputs.logits
                    end_time = time.perf_counter()
                    past_key_values = outputs.past_key_values
                prefill_time = end_time - start_time
            
                # next_token_logits = outputs.logits[:, -1, :]
                # next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                # input_ids = torch.tensor([[next_token_id]]).cuda()

                start_time = time.perf_counter()
                outputs = model.generate(input_ids, past_key_values=past_key_values, return_dict_in_generate=True, 
                            max_new_tokens=args.max_out_len, generation_config=generation_config)
                # for _ in range(args.max_out_len):
                #     outputs = model(input_ids=input_ids, past_key_values=past_key_values)
                #     past_key_values = outputs.past_key_values
                #     next_token_logits = outputs.logits[:, -1, :]
                #     next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                #     input_ids = next_token_id
                end_time = time.perf_counter()
                decode_time = end_time - start_time

            time_list.append([prefill_time, decode_time])

    # try:
    #     # local_rank = int(os.environ["LOCAL_RANK"])
    #     # 下面是 profile 结果的存放路径
    #     p.export_chrome_trace(f"./trace240826_1-wo_empty_cache.json")
    # except Exception as e:
    #     print(f"Failed to capture snapshot {e}!")    

    metrics = {
        "model_path": args.model_path,
        "prefill_time": float(f"{numpy.mean(numpy.array(time_list)[:, 0]):.6f}"),  # pm.duration / min(data_length, 20)
        "decode_time": float(f"{numpy.mean(numpy.array(time_list)[:, 1]):.6f}"),  # pm.duration / min(data_length, 20)
        "memory": float(f"{pm.max_memory:.4f}"),
        "data_length": data_length,
        "input_length": args.input_length,
        "long_cache_config": config.long_cache_config.to_dict() if hasattr(config, "long_cache_config") else None,
    }

    if args.save_file != '':
        with open(args.save_file, "w") as f:
            import json; json.dump(metrics, f, indent=4)

    print(metrics)

if __name__ == "__main__":
    
    main()
        

# torchrun --nnodes=1 --nproc_per_node=1 240824_time_mem_top1.py 

"""
cmd="cd /cpfs01/user/liuxiaoran/projects/activation-beacon-eval/ && /cpfs01/user/liuxiaoran/environment/llm_env_eval_torch2_1/bin/torchrun --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT --nproc_per_node=$num_gpus --nnodes=\$WORLD_SIZE --node_rank=\$RANK eval_cached_ppl.py --model_path=/cpfs01/shared/public/liuxiaoran/llm/llama-2-7b-hf --dataset_fp=/cpfs01/user/liuxiaoran/projects/activation-beacon-eval/data/activation-beacon/lm/pg19.json --input_length=32000 --chunk_length=1024"

/cpfs01/shared/public/llm-env-test/conda_env/python_dlc_env/bin/python /cpfs01/user/liuxiaoran/projects/LongScore/xrliu/run_ali_task.py --job_name internlm2n-0607_llm_score-chenzhi_cn \
    --priority 6 \
    --resource_id=quotau5v1fq37nhb \
    --data_sources=d-pvrngjemoh7p0hb33k,d-wbkgru4hfx0obhg2f0,d-j3ifdf31bulrroktpa,d-o4vq372vln1z2f2zvt,d-3skgd559xp18uawwag \
    --workers 1 \
    --worker_cpu 8 \
    --worker_gpu 1 \
    --worker_memory "100Gi" \
    --worker_image pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/chenxun-st:llm-test \
    --workspace_id 35241 \
    --worker_shared_memory "100Gi" \
    --command "export HOME=/cpfs01/user/liuxiaoran && cd /cpfs01/user/liuxiaoran/projects/activation-beacon-eval/ && /cpfs01/user/liuxiaoran/environment/llm_env_eval_torch2_1/bin/torchrun --master_addr=\$MASTER_ADDR --master_port=\$MASTER_PORT --nproc_per_node=$num_gpus --nnodes=\$WORLD_SIZE --node_rank=\$RANK eval_cached_ppl.py --model_path=/cpfs01/shared/public/liuxiaoran/llm/llama-2-7b-hf --dataset_fp=/cpfs01/user/liuxiaoran/projects/activation-beacon-eval/data/activation-beacon/lm/pg19.json --input_length=32000 --chunk_length=1024" \
    --preemptible true
"""