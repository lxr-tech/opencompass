from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

import torch

with read_base():

    ## longbench

    # from opencompass.configs.datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    # from opencompass.configs.datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    # from opencompass.configs.datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    # from opencompass.configs.datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    # from opencompass.configs.datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    # from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    # from opencompass.configs.datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    # from opencompass.configs.datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    # from opencompass.configs.datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    # from opencompass.configs.datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    # from opencompass.configs.datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    # from opencompass.configs.datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    # from opencompass.configs.datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    # from opencompass.configs.datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    # from opencompass.configs.datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    # from opencompass.configs.datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    # from opencompass.configs.datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    # from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    # from opencompass.configs.datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    # from opencompass.configs.datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    # from opencompass.configs.datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

    # # longbench v2

    # # from opencompass.configs.datasets.longbenchv2.longbenchv2_gen_75fbba import LongBenchv2_datasets
    # from opencompass.configs.datasets.longbenchv2.longbenchv2_gen_no_cot import LongBenchv2_datasets

    ## leval
    
    from opencompass.configs.datasets.leval.levaltpo.leval_tpo_gen import LEval_tpo_datasets
    from opencompass.configs.datasets.leval.levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    from opencompass.configs.datasets.leval.levalquality.leval_quality_gen import LEval_quality_datasets
    from opencompass.configs.datasets.leval.levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    from opencompass.configs.datasets.leval.levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    from opencompass.configs.datasets.leval.levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    
    from opencompass.configs.datasets.leval.levalmultidocqa.leval_multidocqa_gen import LEval_multidocqa_datasets
    from opencompass.configs.datasets.leval.levalpaperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    from opencompass.configs.datasets.leval.levalnaturalquestion.leval_naturalquestion_gen import LEval_nq_datasets
    from opencompass.configs.datasets.leval.levalfinancialqa.leval_financialqa_gen import LEval_financialqa_datasets
    from opencompass.configs.datasets.leval.levallegalcontractqa.leval_legalcontractqa_gen import LEval_legalqa_datasets
    from opencompass.configs.datasets.leval.levalnarrativeqa.leval_narrativeqa_gen import LEval_narrativeqa_datasets

    # # from opencompass.configs.datasets.leval.levalnewssumm.leval_newssumm_gen import LEval_newssumm_datasets
    # # from opencompass.configs.datasets.leval.levalgovreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets
    # # from opencompass.configs.datasets.leval.levalpatentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets
    # # from opencompass.configs.datasets.leval.levaltvshowsumm.leval_tvshow_summ_gen import LEval_tvshow_summ_datasets
    # # from opencompass.configs.datasets.leval.levalmeetingsumm.leval_meetingsumm_gen import LEval_meetingsumm_datasets
    # # from opencompass.configs.datasets.leval.levalreviewsumm.leval_review_summ_gen import LEval_review_summ_datasets

    ## infinitebench
    
    # from opencompass.configs.datasets.infinitebench.infinitebenchcodedebug.infinitebench_codedebug_gen import InfiniteBench_codedebug_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchcoderun.infinitebench_coderun_gen import InfiniteBench_coderun_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchendia.infinitebench_endia_gen import InfiniteBench_endia_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchenmc.infinitebench_enmc_gen import InfiniteBench_enmc_datasets
    from opencompass.configs.datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen import InfiniteBench_enqa_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchmathcalc.infinitebench_mathcalc_gen import InfiniteBench_mathcalc_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchmathfind.infinitebench_mathfind_gen import InfiniteBench_mathfind_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchretrievekv.infinitebench_retrievekv_gen import InfiniteBench_retrievekv_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchretrievenumber.infinitebench_retrievenumber_gen import InfiniteBench_retrievenumber_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen import InfiniteBench_retrievepasskey_datasets
    # from opencompass.configs.datasets.infinitebench.infinitebenchzhqa.infinitebench_zhqa_gen import InfiniteBench_zhqa_datasets



datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    'llama3_2_3b': 1, 'llama3_2_3b_chat': 1, 'llama3_1_8b': 1, 'llama3_1_8b_chat': 1, 

    'qwen3_4b_base': 1, 'qwen3_4b': 1, 'qwen3_8b_base': 1, 'qwen3_8b': 1, 

    'qwen2_5_7b': 1, 'qwen2_5_7b_chat': 1, 'qwen2_5_3b': 1, 'qwen2_5_3b_chat': 1, 
    'qwen2_5_1b': 1, 'qwen2_5_1b_chat': 1, 'qwen2_5_500m': 1, 
    
    'qwen2_5_7b_long': 1, 'qwen2_5_14b_long': 2, 

    'internlm2_5_7b': 1, 'internlm3_8b_chat': 1, 

    'qwen2_5_32b': 4, 'qwen2_5_32b_chat': 4, 'qwq_32b': 4, 'r1_distill_32b': 4, 
}

models = [
    ('llama3_2_3b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/'),
    ('llama3_2_3b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B-Instruct/'),
    ('llama3_1_8b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/'),
    ('llama3_1_8b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B-Instruct/'),

    ('internlm3_8b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/internlm3-8b-instruct/'), 

    ('qwen3_4b_base-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B-Base/'), 
    ('qwen3_4b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B/'), 
    ('qwen3_8b_base-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B-Base/'), 
    ('qwen3_8b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B/'), 

    # ('qwen2_5_500m-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-0.5B/'), 
    # ('qwen2_5_1b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B/'), 
    # ('qwen2_5_1b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-1.5B-Instruct/'), 
    # ('qwen2_5_3b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B/'), 
    # ('qwen2_5_3b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-3B-Instruct/'), 
    # ('qwen2_5_7b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B/'), 
    # ('qwen2_5_7b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct/'), 

    # ('qwen2_5_7b_long-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-7B-Instruct-1M/'), 
    # ('qwen2_5_14b_long-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-14B-Instruct-1M/'), 

    # ('qwen2_5_32b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B/'), 
    # ('qwen2_5_32b_chat-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen2.5-32B-Instruct/'), 
    # ('qwq_32b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/QwQ-32B-Preview/'),
    # # ('r1_distill_32b-32k', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B/'),
]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, drop_middle=True,  # add drop_middle after 250623
        model_kwargs={'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.bfloat16, } if 'c2s' in abbr else {'attn_implementation': 'flash_attention_2'}, 
        max_seq_len=31500, max_out_len=500, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/llm_long/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
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

# source /cpfs01/user/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1
# python run.py eval_xrliu/eval_xrliu_abc_long.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_abc_long.py --dump-eval-details -r 20240820_190019 第一次用
