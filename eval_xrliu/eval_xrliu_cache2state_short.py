from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

import torch

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets

    from opencompass.configs.datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets  # zero shot

    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen_5ddc62 import truthfulqa_datasets  # noqa: F401, F403

    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

    from opencompass.configs.datasets.gpqa.gpqa_ppl_6bf57a import gpqa_datasets
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

batch_size = {
    'llama3_2_3b': 8, 'llama3_1_8b': 8, 'qwen2_5_1b': 8, 'qwen2_5_7b': 8, 
    'qwq_32b': 1, 'r1_distill_32b': 1, 
}

models = [
    ('llama3_2_3b-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    ('llama3_1_8b-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),

    # ('llama3_2_3b-250413-step200-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-pt-baseline/checkpoint-200'), 
    # ('llama3_2_3b-250413-step992-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-pt-baseline/checkpoint-992'), 

    # ('llama3_2_3b-250413-c2s_rfm_sort_14-step992-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-hybrid-sort-pt-LongWanjuan/checkpoint-992'), 
    # ('llama3_2_3b-250413-c2s_rfm_sort2_14-step992-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-hybrid-sort2-pt-LongWanjuan/checkpoint-992'), 
    # ('llama3_2_3b-250413-c2s_rfm_sort2_nf512_14-step992-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-hybrid-sort2f512-pt-LongWanjuan/checkpoint-992'),

    # ('llama3_2_3b-250413-c2s_rfm_sort2_14_rp-step992-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-rp-sort2/checkpoint-992'), 

    ('qwen2_5_1b-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    ('qwen2_5_7b-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 
    ('qwq_32b-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    ('r1_distill_32b-2k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        model_kwargs={'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.bfloat16, } if 'c2s' in abbr else {'attn_implementation': 'flash_attention_2'}, 
        max_seq_len=2048, max_out_len=50, batch_size=batch_size[abbr.split('-')[0]], run_cfg=dict(num_gpus=1),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/cache2state_short/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4, retry=2, 
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16, retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# source /cpfs01/user/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1
# python run.py eval_xrliu/eval_xrliu_cache2state_short.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_cache2state_short.py --dump-eval-details -r 20240820_190019 第一次用
