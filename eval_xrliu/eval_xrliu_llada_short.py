from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

import torch

with read_base():
    # from opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    # from opencompass.configs.datasets.mbpp.deprecated_mbpp_passk_gen_1e1056 import mbpp_datasets
    # from opencompass.configs.datasets.mbpp.deprecated_sanitized_mbpp_passk_gen_1e1056 import sanitized_mbpp_datasets
    # from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    # from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_250710 import humaneval_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    'llama_3_8b_base': 1, 'llama_3_8b_chat': 1,

    'llada_8b_base': 1, 'llada_8b_chat': 1, 'llada_1_5_8b': 1, 

    'dream_v0_7b_base': 1, 'dream_v0_7b_chat': 1, 
}

path_dict = {
    'llama_3_8b_base': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B',
    'llama_3_8b_chat': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B-Instruct/',

    'llada_8b_base': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-8B-Base/', 
    'llada_8b_chat': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-8B-Instruct/', 
    
    'llada_1_5_8b': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-1.5/', 

    'dream_v0_7b_chat': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Dream-v0-Instruct-7B/', 
    'dream_v0_7b_base': '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Dream-v0-Base-7B/', 
}

models = [
    ('llama_3_8b_base-o64-2k', {}, {}, 2048, 512), 
    ('llama_3_8b_chat-o64-2k', {}, {}, 2048, 512), 

    ('llada_8b_base-o512_b32_s512-2k', {}, {'steps': 512, 'block_length': 32, }, 2048, 512), 
    ('llada_8b_chat-o512_b32_s512-2k', {}, {'steps': 512, 'block_length': 32, }, 2048, 512), 

    ('llada_1_5_8b-o512_b32_s512-2k', {}, {'steps': 512, 'block_length': 32, }, 2048, 512), 

    ('dream_v0_7b_base-o512_b32_s512-2k', {}, {'steps': 512, }, 2048, 512), 
    ('dream_v0_7b_chat-o512_b32_s512-2k', {}, {'steps': 512, }, 2048, 512), 
]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], drop_middle=True, 
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1, max_seq_len=max_seq_len,  # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, max_seq_len, max_out_len in models
]

work_dir = './outputs_xrliu/llada_short/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=16),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=4, retry=2, 
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32, retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# source /cpfs01/user/liuxiaoran/.bashrc
# conda activate llm-llada
# python run.py eval_xrliu/eval_xrliu_llada_short.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_llada_short.py --dump-eval-details -r 20240820_190019 第一次用
