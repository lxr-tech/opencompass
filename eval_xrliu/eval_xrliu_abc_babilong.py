from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

with read_base():

    from opencompass.configs.datasets.babilong.babilong_0k_gen import babiLong_0k_datasets
    from opencompass.configs.datasets.babilong.babilong_2k_gen import babiLong_2k_datasets
    from opencompass.configs.datasets.babilong.babilong_4k_gen import babiLong_4k_datasets
    from opencompass.configs.datasets.babilong.babilong_8k_gen import babiLong_8k_datasets
    from opencompass.configs.datasets.babilong.babilong_16k_gen import babiLong_16k_datasets
    from opencompass.configs.datasets.babilong.babilong_32k_gen import babiLong_32k_datasets
    from opencompass.configs.datasets.babilong.babilong_64k_gen import babiLong_64k_datasets
    from opencompass.configs.datasets.babilong.babilong_128k_gen import babiLong_128k_datasets

datasets = []

datasets += babiLong_0k_datasets
datasets += babiLong_2k_datasets
datasets += babiLong_4k_datasets
datasets += babiLong_8k_datasets
datasets += babiLong_16k_datasets
datasets += babiLong_32k_datasets
# datasets += babiLong_64k_datasets
# datasets += babiLong_128k_datasets

num_gpus = {
    'llama3_8b': 1, 'llama3_8b_chat': 1, 

    'llama3_1_8b': 1, 'llama3_1_8b_chat': 1, 'llama3_2_3b': 1, 'llama3_2_3b_chat': 1, 
    
    'qwen3_4b_base': 1, 'qwen3_4b': 1, 'qwen3_8b_base': 1, 'qwen3_8b': 1, 

    'qwen2_5_7b': 1, 'qwen2_5_7b_chat': 1, 'qwen2_5_3b': 1, 'qwen2_5_3b_chat': 1, 
    'qwen2_5_1b': 1, 'qwen2_5_1b_chat': 1, 'qwen2_5_500m': 1, 
    
    'qwen2_5_7b_long': 1, 'qwen2_5_14b_long': 2, 

    'internlm2_5_7b': 1, 'internlm3_8b_chat': 1, 

    'qwen2_5_32b': 4, 'qwen2_5_32b_chat': 4, 'qwq_32b': 4, 'r1_distill_32b': 4, 
}

models = [
    # ('llama3_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B'),
    # ('llama3_8b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B-Instruct/'),

    ('llama3_2_3b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    ('llama3_2_3b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B-Instruct/'),
    ('llama3_1_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),
    ('llama3_1_8b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B-Instruct/'),

    ('internlm3_8b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/internlm3-8b-instruct/'), 

    ('qwen3_4b_base', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B-Base/'), 
    ('qwen3_4b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B/'), 
    ('qwen3_8b_base', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B-Base/'), 
    ('qwen3_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B/'), 

    # ('qwen2_5_500m', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-0.5B/'), 
    # ('qwen2_5_1b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    # ('qwen2_5_1b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B-Instruct/'), 
    # ('qwen2_5_3b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B/'), 
    # ('qwen2_5_3b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B-Instruct/'), 
    # ('qwen2_5_7b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 
    # ('qwen2_5_7b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct/'), 

    # ('qwen2_5_7b_long', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct-1M/'), 
    # ('qwen2_5_14b_long', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-14B-Instruct-1M/'), 

    # ('qwen2_5_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B/'), 
    # ('qwen2_5_32b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B-Instruct/'), 
    # ('qwq_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    # ('r1_distill_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        model_kwargs={'attn_implementation': 'flash_attention_2'}, 
        max_out_len=64, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/llm_babilong/'

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
        max_num_workers=64, 
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
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py eval_xrliu/eval_xrliu_abc_babilong.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_abc_babilong.py --dump-eval-details -r 20240820_190019 第一次用
