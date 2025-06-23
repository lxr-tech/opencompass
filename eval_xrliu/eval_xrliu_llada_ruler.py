from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    # from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k

datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
# datasets += ruler_datasets_32k

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

""" code for 20250606_131127
    models = [
        ('llada_8b_base', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-8B-Base/', 
        {}),
        ('llada_8b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-8B-Instruct/', 
        {}),

        ('llada_8b_base-ntk4', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-8B-Base/', 
        {'scaling_factor': 4}),
        ('llada_8b_base-ntk13', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-8B-Base/', 
        {'scaling_factor': 13}),
        
        # ('llada_1_5_8b_base', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/LLaDA-1.5/'),
    ]

    models = [
        dict(
            type=LLaDACausalLM, abbr=abbr, path=path, 
            model_kwargs={'flash_attention': True}, scaling_config=scaling_config, 
            max_out_len=64, batch_size=1, 
            run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
        ) for abbr, path, scaling_config in models
    ]
"""

models = [

    # ('llama_3_8b_base-o64', {}, {}, 64), 
    # ('llama_3_8b_base-o64-ntk4', {'scaling_factor': 4}, {}, 64), 
    # ('llama_3_8b_base-o64-ntk13', {'scaling_factor': 13}, {}, 64), 

    # ('llama_3_8b_chat-o64', {}, {}, 64), 
    # ('llama_3_8b_chat-o64-ntk4', {'scaling_factor': 4}, {}, 64), 
    # ('llama_3_8b_chat-o64-ntk13', {'scaling_factor': 13}, {}, 64), 

    # ('llada_8b_base-o64_b64_s64', {}, {'steps': 64, 'block_length': 64, }, 64), 

    # ('llada_8b_base-o64_b64_s64-ntk4', {'scaling_factor': 4}, {'steps': 64, 'block_length': 64, }, 64), 
    # ('llada_8b_base-o64_b64_s64-ntk14', {'scaling_factor': 14}, {'steps': 64, 'block_length': 64, }, 64), 
    # ('llada_8b_base-o64_b64_s64-ntk31', {'scaling_factor': 31}, {'steps': 64, 'block_length': 64, }, 64), 

    ('llada_8b_chat-o64_b64_s64', {}, {'steps': 64, 'block_length': 64, }, 64), 

    ('llada_8b_chat-o64_b64_s64-ntk4', {'scaling_factor': 4}, {'steps': 64, 'block_length': 64, }, 64), 
    ('llada_8b_chat-o64_b64_s64-ntk14', {'scaling_factor': 14}, {'steps': 64, 'block_length': 64, }, 64), 
    ('llada_8b_chat-o64_b64_s64-ntk31', {'scaling_factor': 31}, {'steps': 64, 'block_length': 64, }, 64), 

    # ('llada_1_5_8b-o64_b64_s64', {}, {'steps': 64, 'block_length': 64, }, 64), 

    # ('llada_1_5_8b-o64_b64_s64-ntk4', {'scaling_factor': 4}, {'steps': 64, 'block_length': 64, }, 64), 
    # ('llada_1_5_8b-o64_b64_s64-ntk14', {'scaling_factor': 14}, {'steps': 64, 'block_length': 64, }, 64), 
    # ('llada_1_5_8b-o64_b64_s64-ntk31', {'scaling_factor': 31}, {'steps': 64, 'block_length': 64, }, 64), 

    # ('dream_v0_7b_base-o64_s64', {}, {'steps': 64, }, 64), 

    # ('dream_v0_7b_base-o64_s64-ntk5', {'scaling_factor': 5}, {'steps': 64, }, 64), 

    # ('dream_v0_7b_chat-o64_s64', {}, {'steps': 64, }, 64), 
    
    # ('dream_v0_7b_chat-o64_s64-ntk5', {'scaling_factor': 5}, {'steps': 64, }, 64), 

]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], 
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, max_out_len in models
]

work_dir = './outputs_xrliu/llada_ruler/'

# File "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py", line 140, in inference
#     entry, golds = list(zip(*datum))
# ValueError: too many values to unpack (expected 2)

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=4, 
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

# summarizer = dict(
#     dataset_abbrs=['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k', 'ruler_128k'],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []
#     ),
# )
 
# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate llm-llada
# python run.py eval_xrliu/eval_xrliu_llada_ruler.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_llada_ruler.py --dump-eval-details -r 20240820_190019 第一次用
