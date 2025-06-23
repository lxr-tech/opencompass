from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

import torch

with read_base():

    # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import mmlu_pro_datasets

    # Math
    # from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen import math_datasets
    # # FileNotFoundError: [Errno 2] No such file or directory: './data/math/test_prm800k_500.json'
    # from opencompass.configs.datasets.aime2024.aime2024_0shot_nocot_genericllmeval_academic_gen import  aime2024_datasets
    # # AssertionError: No valid url for ./data/aime.jsonl!

    # General Reasoning
    # from opencompass.configs.datasets.bbh.bbh_0shot_nocot_academic_gen import bbh_datasets
    from opencompass.configs.datasets.gpqa.gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets

    # Instruction Following
    from opencompass.configs.datasets.IFEval.IFEval_gen_353ae7 import ifeval_datasets

    # from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import humaneval_datasets
    # from opencompass.configs.datasets.livecodebench.livecodebench_gen_a4f90b import LCBCodeGeneration_dataset

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    'llama3_2_3b': 1, 'llama3_2_3b_chat': 1, 'llama3_1_8b': 1, 'llama3_1_8b_chat': 1, 

    'qwen3_4b_base': 1, 'qwen3_4b': 1, 'qwen3_8b_base': 1, 'qwen3_8b': 1, 

    'qwen2_5_7b': 1, 'qwen2_5_7b_chat': 1, 'qwen2_5_3b': 1, 'qwen2_5_3b_chat': 1, 
    'qwen2_5_1b': 1, 'qwen2_5_1b_chat': 1, 'qwen2_5_500m': 1, 

    'qwen2_5_14b_chat' : 2, 
    
    'qwen2_5_7b_long': 1, 'qwen2_5_14b_long': 2, 

    'internlm2_5_7b': 1, 'internlm3_8b_chat': 1, 

    'qwen2_5_32b': 4, 'qwen2_5_32b_chat': 4, 'qwq_32b': 4, 'r1_distill_32b': 4, 
}

models = [
    # ('llama3_2_3b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    # ('llama3_2_3b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B-Instruct/'),
    # ('llama3_1_8b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),
    # ('llama3_1_8b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B-Instruct/'),

    # ('internlm3_8b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/internlm3-8b-instruct/'), 

    # ('qwen3_4b_base-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B-Base/'), 
    # ('qwen3_4b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B/'), 
    # ('qwen3_8b_base-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B-Base/'), 
    # ('qwen3_8b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B/'), 

    # ('qwen2_5_500m-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-0.5B/'), 
    # ('qwen2_5_1b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    # ('qwen2_5_1b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B-Instruct/'), 
    # ('qwen2_5_3b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B/'), 
    # ('qwen2_5_3b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B-Instruct/'), 
    # ('qwen2_5_7b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 
    # ('qwen2_5_7b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct/'), 
    ('qwen2_5_14b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-14B-Instruct/'), 

    # ('qwen2_5_7b_long-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct-1M/'), 
    ('qwen2_5_14b_long-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-14B-Instruct-1M/'), 

    # # ('qwen2_5_32b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B/'), 
    # ('qwen2_5_32b_chat-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B-Instruct'), 
    # ('qwq_32b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    # # ('r1_distill_32b-16k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        model_kwargs={'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.bfloat16, } if 'c2s' in abbr else {'attn_implementation': 'flash_attention_2'}, 
        max_seq_len=16384, max_out_len=4096, batch_size=2, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/cache2state_lcot/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=2,
        # retry=2, 
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# source /cpfs01/user/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1
# python run.py eval_xrliu/eval_xrliu_cache2state_lcot.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_cache2state_lcot.py --dump-eval-details -r 20240820_190019 第一次用
