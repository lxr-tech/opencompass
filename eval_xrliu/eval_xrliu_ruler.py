from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k

datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
datasets += ruler_datasets_32k

num_gpus = {
    'llama3_2_3b': 1, 'llama3_1_8b': 1, 
    'qwen2_5_32b': 4, 'qwen2_5_7b': 1, 'qwen2_5_3b': 1, 'qwen2_5_1b': 1, 'qwen2_5_500m': 1, 
    'qwq_32b': 4, 'r1_distill_32b': 4, 
}

models = [
    # ('llama3_2_3b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    # ('llama3_1_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),

    ('qwen2_5_500m', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-0.5B/'), 
    # ('qwen2_5_1b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    # ('qwen2_5_3b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B/'), 
    # ('qwen2_5_7b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 

    # ('qwen2_5_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B/'), 
    # ('qwq_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    # ('r1_distill_32b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        model_kwargs={'attn_implementation': 'flash_attention_2'}, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/ruler/'

#   File "/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/opencompass/opencompass/openicl/icl_inferencer/icl_gen_inferencer.py", line 140, in inference
#     entry, golds = list(zip(*datum))
# ValueError: too many values to unpack (expected 2)

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8, 
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
# python run.py eval_xrliu/eval_xrliu_ruler.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_ruler.py --dump-eval-details -r 20240820_190019 第一次用
