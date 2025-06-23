from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)

num_gpus = {
    'llama3_8b': 1, 'llama3_8b_chat': 1, 

    'llama3_1_8b': 1, 'llama3_1_8b_chat': 1, 'llama3_2_3b': 1, 'llama3_2_3b_chat': 1, 

    'qwen3_4b_base': 1, 'qwen3_4b': 1, 'qwen3_8b_base': 1, 'qwen3_8b': 1, 

    'qwen2_5_7b': 1, 'qwen2_5_7b_chat': 1, 'qwen2_5_3b': 1, 'qwen2_5_3b_chat': 1, 
    'qwen2_5_1b': 1, 'qwen2_5_1b_chat': 1, 'qwen2_5_500m': 1, 

    'qwen2_5_32b': 4, 'qwen2_5_32b_chat': 4, 'qwq_32b': 4, 'r1_distill_32b': 4, 
}

models = [
    ('llama3_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B'),
    ('llama3_8b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B-Instruct/'),

    # ('llama3_2_3b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    # ('llama3_2_3b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B-Instruct/'),
    # ('llama3_1_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),
    # ('llama3_1_8b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B-Instruct/'),

    # ('qwen3_4b_base', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B-Base/'), 
    # ('qwen3_4b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B/'), 
    # ('qwen3_8b_base', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B-Base/'), 
    # ('qwen3_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B/'), 

    # ('qwen2_5_500m', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-0.5B/'), 
    # ('qwen2_5_1b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    # ('qwen2_5_1b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B-Instruct/'), 
    # ('qwen2_5_3b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B/'), 
    # ('qwen2_5_3b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B-Instruct/'), 
    # ('qwen2_5_7b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 
    # ('qwen2_5_7b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct/'), 

    # ('qwen2_5_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B/'), 
    # ('qwen2_5_32b_chat', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B-Instruct/'), 
    # ('qwq_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    # ('r1_distill_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        model_kwargs={'attn_implementation': 'flash_attention_2'}, 
        max_out_len=50 if is_single_niah else 250, batch_size=1, run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/niah/'

# File "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py", line 140, in inference
#     entry, golds = list(zip(*datum))
# ValueError: too many values to unpack (expected 2)

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=2, 
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16, 
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
# python run.py eval_xrliu/eval_xrliu_niah.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_niah.py --dump-eval-details -r 20240820_190019 第一次用
