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

"""
    >>> 2 * np.ceil(128 / 2 * np.log(4096 / 2 / np.pi) / np.log(500000))
    64.0
    >>> (8192 / 2 / np.pi) ** (128 / 64) / 500000
    3.3997747666863356
    >>> (16384 / 2 / np.pi) ** (128 / 64) / 500000
    13.599099066745342
    >>> 2 * np.ceil(128 / 2 * np.log(2048 / 2 / np.pi) / np.log(1000000))
    54.0
    >>> (4096 / 2 / np.pi) ** (128 / 54) / 1000000
    4.684347204817047
    >>> (8192 / 2 / np.pi) ** (128 / 54) / 1000000
    24.221534862197895
"""

#  20250606_131127: 
#   no random_seed, max_out_len=64, 
#   default: {'steps': 128, 'block_length': 32, 'temperature': 0., 'cfg_scale': 0., 'remasking': 'low_confidence'}

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
    # # ('llada_8b_base-o32_b32_s64', {}, {'steps': 64, 'block_length': 32, }, 32), 
    # # ('llada_8b_base-o32_b32_s128', {}, {'steps': 128, 'block_length': 32, }, 32), 

    # ('llada_8b_base-o128_b128_s1', {}, {'steps': 1, 'block_length': 128, }, 128), 
    # ('llada_8b_base-o128_b128_s4', {}, {'steps': 4, 'block_length': 128, }, 128), 
    # ('llada_8b_base-o128_b128_s8', {}, {'steps': 8, 'block_length': 128, }, 128), 
    # ('llada_8b_base-o128_b128_s16', {}, {'steps': 16, 'block_length': 128, }, 128), 
    # ('llada_8b_base-o128_b128_s32', {}, {'steps': 32, 'block_length': 128, }, 128), 
    # ('llada_8b_base-o128_b128_s64', {}, {'steps': 64, 'block_length': 128, }, 128), 
    # ('llada_8b_base-o128_b128_s128', {}, {'steps': 128, 'block_length': 128, }, 128), 

    # ('llada_8b_chat-o32_b32_s1', {}, {'steps': 1, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s2', {}, {'steps': 2, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s4', {}, {'steps': 4, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s8', {}, {'steps': 8, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s16', {}, {'steps': 16, 'block_length': 32, }, 32), 
    # ('llada_8b_chat-o32_b32_s32', {}, {'steps': 32, 'block_length': 32, }, 32), 
    # # ('llada_8b_chat-o32_b32_s64', {}, {'steps': 64, 'block_length': 32, }, 32), 
    # # ('llada_8b_chat-o32_b32_s128', {}, {'steps': 128, 'block_length': 32, }, 32), 
 
    # ('llada_8b_base-o32_b32_s32-ntk4', {'scaling_factor': 4}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk14', {'scaling_factor': 14}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk31', {'scaling_factor': 31}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_base-o32_b32_s32-ntk55', {'scaling_factor': 55}, {'steps': 32, 'block_length': 32, }, 32),

    # ('llada_8b_chat-o32_b32_s32-ntk4', {'scaling_factor': 4}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_chat-o32_b32_s32-ntk14', {'scaling_factor': 14}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_chat-o32_b32_s32-ntk31', {'scaling_factor': 31}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_8b_chat-o32_b32_s32-ntk55', {'scaling_factor': 55}, {'steps': 32, 'block_length': 32, }, 32),

    ('llada_1_5_8b-o32_b32_s1', {}, {'steps': 1, 'block_length': 32, }, 32), 
    ('llada_1_5_8b-o32_b32_s2', {}, {'steps': 2, 'block_length': 32, }, 32), 
    ('llada_1_5_8b-o32_b32_s4', {}, {'steps': 4, 'block_length': 32, }, 32), 
    ('llada_1_5_8b-o32_b32_s8', {}, {'steps': 8, 'block_length': 32, }, 32), 
    ('llada_1_5_8b-o32_b32_s16', {}, {'steps': 16, 'block_length': 32, }, 32), 
    ('llada_1_5_8b-o32_b32_s32', {}, {'steps': 32, 'block_length': 32, }, 32), 

    # ('llada_1_5_8b-o32_b32_s32-ntk4', {'scaling_factor': 4}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_1_5_8b-o32_b32_s32-ntk14', {'scaling_factor': 14}, {'steps': 32, 'block_length': 32, }, 32),
    # ('llada_1_5_8b-o32_b32_s32-ntk31', {'scaling_factor': 31}, {'steps': 32, 'block_length': 32, }, 32),

    # ('dream_v0_7b_base-o32_s1', {}, {'steps': 1, }, 32), 
    # ('dream_v0_7b_base-o32_s8', {}, {'steps': 8, }, 32), 
    # ('dream_v0_7b_base-o32_s16', {}, {'steps': 16, }, 32), 
    # ('dream_v0_7b_base-o32_s32', {}, {'steps': 32, }, 32), 
 
    # ('dream_v0_7b_base-o32_s32-ntk5', {'scaling_factor': 5}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_base-o32_s32-ntk25', {'scaling_factor': 25}, {'steps': 32, }, 32), 
    # ('dream_v0_7b_base-o32_s32-ntk126', {'scaling_factor': 126}, {'steps': 32, }, 32), 

    # ('dream_v0_7b_chat-o32_s1', {}, {'steps': 1, }, 32), 
    # ('dream_v0_7b_chat-o32_s8', {}, {'steps': 8, }, 32), 
    # ('dream_v0_7b_chat-o32_s16', {}, {'steps': 16, }, 32), 
    # ('dream_v0_7b_chat-o32_s32', {}, {'steps': 32, }, 32), 
 
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
