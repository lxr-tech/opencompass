from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)

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
    # ('llama_3_8b_base-o32', {}, {}, 32), 
    # ('llama_3_8b_base-o32-ntk4', {'scaling_factor': 4}, {}, 32), 
    # ('llama_3_8b_base-o32-ntk13', {'scaling_factor': 13}, {}, 32), 
    
    # ('llama_3_8b_chat-o32', {}, {}, 32), 
    # ('llama_3_8b_chat-o32-ntk4', {'scaling_factor': 4}, {}, 32), 
    # ('llama_3_8b_chat-o32-ntk13', {'scaling_factor': 13}, {}, 32), 

    # ('llada_8b_base-o32_b32_s1', {}, {'steps': 1, 'block_length': 32, }, 32), 
    # ('llada_8b_base-o32_b32_s2', {}, {'steps': 2, 'block_length': 32, }, 32), 
    # ('llada_8b_base-o32_b32_s4', {}, {'steps': 4, 'block_length': 32, }, 32), 
    # ('llada_8b_base-o32_b32_s8', {}, {'steps': 8, 'block_length': 32, }, 32), 
    # ('llada_8b_base-o32_b32_s16', {}, {'steps': 16, 'block_length': 32, }, 32), 
    # ('llada_8b_base-o32_b32_s32', {}, {'steps': 32, 'block_length': 32, }, 32), 

    # ('llada_8b_chat-o32_b32_s1', {}, {'steps': 1, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s2', {}, {'steps': 2, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s4', {}, {'steps': 4, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s8', {}, {'steps': 8, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s16', {}, {'steps': 16, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s32', {}, {'steps': 32, 'block_length': 32, }, 32), 

    # ('llada_1_5_8b-o32_b32_s1', {}, {'steps': 1, 'block_length': 32, }, 32), 
    # ('llada_1_5_8b-o32_b32_s2', {}, {'steps': 2, 'block_length': 32, }, 32), 
    # ('llada_1_5_8b-o32_b32_s4', {}, {'steps': 4, 'block_length': 32, }, 32), 
    # ('llada_1_5_8b-o32_b32_s8', {}, {'steps': 8, 'block_length': 32, }, 32), 
    # ('llada_1_5_8b-o32_b32_s16', {}, {'steps': 16, 'block_length': 32, }, 32), 
    # ('llada_1_5_8b-o32_b32_s32', {}, {'steps': 32, 'block_length': 32, }, 32), 
 
    # ('llada_8b_base-o32_b32_s32-ntk4', {'scaling_factor': 4}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk14', {'scaling_factor': 14}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk31', {'scaling_factor': 31}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk55', {'scaling_factor': 55}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk_dyn', {'scaling_factor': -1, }, {'steps': 32, 'block_length': 32, }, 32),

    # # ('llada_8b_base-o32_b32_s32-ntk31-log_auto', {'scaling_factor': 31, 'log_scale': -1}, {'steps': 32, 'block_length': 32, }, 32),
    # # ('llada_8b_base-o32_b32_s32-ntk55-log_auto', {'scaling_factor': 55, 'log_scale': -1}, {'steps': 32, 'block_length': 32, }, 32),
    # # ('llada_8b_base-o32_b32_s32-ntk_dyn-log_auto', {'scaling_factor': -1, 'log_scale': -1}, {'steps': 32, 'block_length': 32, }, 32),

    # ('llada_8b_chat-o32_b32_s32-ntk4', {'scaling_factor': 4}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_chat-o32_b32_s32-ntk14', {'scaling_factor': 14}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_chat-o32_b32_s32-ntk31', {'scaling_factor': 31}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_chat-o32_b32_s32-ntk55', {'scaling_factor': 55}, {'steps': 32, 'block_length': 32, }, 32),

    # ('llada_1_5_8b-o32_b32_s32-ntk4', {'scaling_factor': 4}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_1_5_8b-o32_b32_s32-ntk14', {'scaling_factor': 14}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_1_5_8b-o32_b32_s32-ntk31', {'scaling_factor': 31}, {'steps': 32, 'block_length': 32, }, 32),

    ('dream_v0_7b_base-o32_s1-de', {}, {'steps': 1, }, 32), 
    ('dream_v0_7b_base-o32_s2-de', {}, {'steps': 2, }, 32), 
    ('dream_v0_7b_base-o32_s4-de', {}, {'steps': 4, }, 32), 
    ('dream_v0_7b_base-o32_s8-de', {}, {'steps': 8, }, 32), 
    ('dream_v0_7b_base-o32_s16-de', {}, {'steps': 16, }, 32), 
    ('dream_v0_7b_base-o32_s32-de', {}, {'steps': 32, }, 32), 
 
    # ('dream_v0_7b_base-o32_s32-de-ntk2', {'scaling_factor': 2}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_base-o32_s32-de-ntk5', {'scaling_factor': 5}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_base-o32_s32-de-ntk7', {'scaling_factor': 7}, {'steps': 32, }, 32), 

    # ('dream_v0_7b_base-o32_s32-ntk5', {'scaling_factor': 5}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_base-o32_s32-ntk25', {'scaling_factor': 25}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_base-o32_s32-ntk126', {'scaling_factor': 126}, {'steps': 32, }, 32), 

    ('dream_v0_7b_chat-o32_s1-de', {}, {'steps': 1, }, 32), 
    ('dream_v0_7b_chat-o32_s2-de', {}, {'steps': 2, }, 32), 
    ('dream_v0_7b_chat-o32_s4-de', {}, {'steps': 4, }, 32), 
    ('dream_v0_7b_chat-o32_s8-de', {}, {'steps': 8, }, 32), 
    ('dream_v0_7b_chat-o32_s16-de', {}, {'steps': 16, }, 32), 
    ('dream_v0_7b_chat-o32_s32-de', {}, {'steps': 32, }, 32), 

    # ('dream_v0_7b_chat-o32_s32-de-ntk2', {'scaling_factor': 2}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_chat-o32_s32-de-ntk5', {'scaling_factor': 5}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_chat-o32_s32-de-ntk7', {'scaling_factor': 7}, {'steps': 32, }, 32), 

    # ('dream_v0_7b_chat-o32_s32-ntk5', {'scaling_factor': 5}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_chat-o32_s32-ntk25', {'scaling_factor': 25}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_chat-o32_s32-ntk126', {'scaling_factor': 126}, {'steps': 32, }, 32), 

]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], 
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, max_out_len in models
]


work_dir = './outputs_xrliu/llada_niah/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py eval_xrliu/eval_xrliu_llada_niah.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_llada_niah.py --dump-eval-details -r 20240820_190019 第一次用
