from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

import torch

with read_base():

    # longbench

    from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

    # longbench v2

    # from opencompass.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen_no_cot import LongBenchv2_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    'llama3_2_3b': 1, 'llama3_1_8b': 1, 
    'qwen2_5_32b': 4, 'qwen2_5_7b': 1, 'qwen2_5_3b': 1, 'qwen2_5_1b': 1, 'qwen2_5_500m': 1, 
    'qwq_32b': 4, 'r1_distill_32b': 4, 
}

models = [
    ('llama3_2_3b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    ('llama3_1_8b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),

    # ('llama3_2_3b-250413-step200-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-pt-baseline/checkpoint-200'), 
    # ('llama3_2_3b-250413-step992-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-pt-baseline/checkpoint-992'), 

    # ('llama3_2_3b-250413-c2s_rfm_sort_14-step992-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-hybrid-sort-pt-LongWanjuan/checkpoint-992'), 
    # ('llama3_2_3b-250413-c2s_rfm_sort2_14-step992-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-hybrid-sort2-pt-LongWanjuan/checkpoint-992'),
    # ('llama3_2_3b-250413-c2s_rfm_sort2_nf512_14-step992-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-hybrid-sort2f512-pt-LongWanjuan/checkpoint-992'),

    # ('llama3_2_3b-250413-c2s_rfm_sort2_14_rp-step992-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/saves/llama3.2-3B-fla-rp-sort2/checkpoint-992'), 

    ('qwen2_5_500m-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-0.5B/'), 
    ('qwen2_5_1b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    ('qwen2_5_3b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B/'), 
    ('qwen2_5_7b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 
    # ('qwen2_5_32b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B/'), 
    # ('qwq_32b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    # ('r1_distill_32b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        model_kwargs={'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.bfloat16, } if 'c2s' in abbr else {'attn_implementation': 'flash_attention_2'}, 
        max_seq_len=31500, max_out_len=500, batch_size=1, run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/cache2state_long/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2, retry=2, 
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
# python run.py eval_xrliu/eval_xrliu_cache2state_long.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_cache2state_long.py --dump-eval-details -r 20240820_190019 第一次用
