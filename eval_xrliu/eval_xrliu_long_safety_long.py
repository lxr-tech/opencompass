from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import DLCRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceBaseModel

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

    # # leval
    
    # from opencompass.configs.datasets.leval.levaltpo.leval_tpo_gen import LEval_tpo_datasets
    # from opencompass.configs.datasets.leval.levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    # from opencompass.configs.datasets.leval.levalquality.leval_quality_gen import LEval_quality_datasets
    # from opencompass.configs.datasets.leval.levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    # from opencompass.configs.datasets.leval.levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    # from opencompass.configs.datasets.leval.levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    
    # from opencompass.configs.datasets.leval.levalmultidocqa.leval_multidocqa_gen import LEval_multidocqa_datasets
    # from opencompass.configs.datasets.leval.levalpaperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets
    # from opencompass.configs.datasets.leval.levalnaturalquestion.leval_naturalquestion_gen import LEval_nq_datasets
    # from opencompass.configs.datasets.leval.levalfinancialqa.leval_financialqa_gen import LEval_financialqa_datasets
    # from opencompass.configs.datasets.leval.levallegalcontractqa.leval_legalcontractqa_gen import LEval_legalqa_datasets
    # from opencompass.configs.datasets.leval.levalnarrativeqa.leval_narrativeqa_gen import LEval_narrativeqa_datasets

    # from opencompass.configs.datasets.leval.levalnewssumm.leval_newssumm_gen import LEval_newssumm_datasets
    # from opencompass.configs.datasets.leval.levalgovreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets
    # from opencompass.configs.datasets.leval.levalpatentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets
    # from opencompass.configs.datasets.leval.levaltvshowsumm.leval_tvshow_summ_gen import LEval_tvshow_summ_datasets
    # from opencompass.configs.datasets.leval.levalmeetingsumm.leval_meetingsumm_gen import LEval_meetingsumm_datasets
    # from opencompass.configs.datasets.leval.levalreviewsumm.leval_review_summ_gen import LEval_review_summ_datasets

    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen_no_cot import LongBenchv2_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

meta_dict = {
    'llama3_1_8b_chat': '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/', 
    'llama3_2_3b_chat': '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72/', 
}

models = [
    # ('llama3_1_8b_chat-32k', '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/'), 
    # ('llama3_1_8b_chat-long_safety_250129-400-32k-de', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250129/400/'), 
    # ('llama3_1_8b_chat-long_safety_250129-1000-32k-de', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250129/1000/'), 

    ('llama3_1_8b_chat-long_safety_250129_64k-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250129-64k/400/'), 
    ('llama3_1_8b_chat-long_safety_250129_64k-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250129-64k/1000/'), 

    # ('llama3_1_8b_chat-long_safety_250129_128k-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250129-128k/400/'), 
    # ('llama3_1_8b_chat-long_safety_250129_128k-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250129-128k/1000/'), 

    # ('llama3_2_3b_chat-32k', '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72/'), 
    # ('llama3_2_3b_chat-long_safety_250129-400-32k-de', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_2_3b-250129/400/'), 
    # ('llama3_2_3b_chat-long_safety_250129-1000-32k-de', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_2_3b-250129/1000/'), 

    # ('llama3_2_3b_chat-long_safety_250129_128k-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_2_3b-250129-128k/400/'), 
    # ('llama3_2_3b_chat-long_safety_250129_128k-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_2_3b-250129-128k/1000/'), 
    
    # ('qwen2_5_7b_chat-long_align_mix_long_safe_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-qwen2_5_7b-250205/400/'), 
    # ('qwen2_5_7b_chat-long_align_mix_long_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-qwen2_5_7b-250205/1000/'), 
    # ('qwen2_5_7b_chat-long_align_mix_short_safe_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_short_safe-qwen2_5_7b-250205/400/'), 
    # ('qwen2_5_7b_chat-long_align_mix_short_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_short_safe-qwen2_5_7b-250205/1000/'), 
    # ('qwen2_5_7b_chat-long_align_mix_hh_rlhf_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_hh_rlhf-qwen2_5_7b-250205/400/'), 
    # ('qwen2_5_7b_chat-long_align_mix_hh_rlhf_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_hh_rlhf-qwen2_5_7b-250205/1000/'), 
    # ('qwen2_5_7b_chat-long_align_mix_beaver_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_beaver-qwen2_5_7b-250205/1000/'), 

    # ('llama3_1_8b_chat-long_align_mix_long_safe_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-llama3_1_8b-250205/400/'), 
    # ('llama3_1_8b_chat-long_align_mix_long_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-llama3_1_8b-250205/1000/'), 
    # ('llama3_1_8b_chat-long_align_mix_short_safe_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_short_safe-llama3_1_8b-250205/400/'), 
    # ('llama3_1_8b_chat-long_align_mix_short_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_short_safe-llama3_1_8b-250205/1000/'), 
    # ('llama3_1_8b_chat-long_align_mix_hh_rlhf_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_hh_rlhf-llama3_1_8b-250205/400/'), 
    # ('llama3_1_8b_chat-long_align_mix_hh_rlhf_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_hh_rlhf-llama3_1_8b-250205/1000/'), 
    # ('llama3_1_8b_chat-long_align_mix_beaver_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_beaver-llama3_1_8b-250205/1000/'), 

    # ('llama3_2_3b_chat-long_align_mix_long_safe_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-llama3_2_3b-250205/400/'), 
    # ('llama3_2_3b_chat-long_align_mix_long_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-llama3_2_3b-250205/1000/'), 
    # ('llama3_2_3b_chat-long_align_mix_short_safe_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_short_safe-llama3_2_3b-250205/400/'), 
    # ('llama3_2_3b_chat-long_align_mix_short_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_short_safe-llama3_2_3b-250205/1000/'), 
    # ('llama3_2_3b_chat-long_align_mix_hh_rlhf_250205-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_hh_rlhf-llama3_2_3b-250205/400/'), 
    # ('llama3_2_3b_chat-long_align_mix_hh_rlhf_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_hh_rlhf-llama3_2_3b-250205/1000/'), 

    # ('qwen2_5_7b_chat-32k', '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75_no_yarn/'), 
    # ('qwen2_5_7b_chat-long_safety_250129-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_7b-250129/400/'), 
    # ('qwen2_5_7b_chat-long_safety_250129-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_7b-250129/1000/'), 

    # ('qwen2_5_7b_chat-long_align_mix_long_safe_250205-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_align_mix_long_safe-qwen2_5_7b-250205/1000/'), 

    # ('qwen2_5_1b_chat-32k', '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/2fd50615a2a9792d223eba8e0741aa90ef21a869/'), 
    # ('qwen2_5_1b_chat-long_safety_250129-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_1b-250129/400/'), 
    # ('qwen2_5_1b_chat-long_safety_250129-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_1b-250129/1000/'), 

    # ('internlm2_5_7b_chat-32k', '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-7b-chat-1m/snapshots/846dee6fb6f5a96fb9ab7d3f1f9c383ac9e73bc1/'), 
    # ('internlm2_5_7b_chat-long_safety_250129-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm2_5_7b-250129/400/'), 
    # ('internlm2_5_7b_chat-long_safety_250129-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm2_5_7b-250129/1000/'), 

    # ('internlm2_5_1b_chat-32k', '/cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-1_8b-chat/snapshots/763507996f322ce22c3e08ed22737fb63ef6613c/'), 
    # ('internlm2_5_1b_chat-long_safety_250129-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm2_5_1b-250129/400/'), 
    # ('internlm2_5_1b_chat-long_safety_250129-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm2_5_1b-250129/1000/'), 

    # ('internlm3_8b_chat-32k', '/cpfs01/shared/public/lvhaijun/develop_internlm3_open_source_hf_01115/20250109095225_hf-080_open_source_hf/'), 
    # ('internlm3_8b_chat-long_safety_250129-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-internlm3_8b-250129/1000'), 

    # ('internlm3_8b_chat-long_rush_250306-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250306/400/'),
    # ('internlm3_8b_chat-long_rush_250306-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250306/1000/'),
    # ('internlm3_8b_chat-long_rush_250307-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250307/400/'), 
    # ('internlm3_8b_chat-long_rush_250307-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250307/1000/'), 

    # ('internlm3_8b_chat-long_rush_250307_32k_code-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250307_32k_code/400/'), 
    # ('internlm3_8b_chat-long_rush_250307_32k_code-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250307_32k_code/1000/'), 

    # ('internlm3_8b_chat-long_rush_250308_32k_code-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250308_32k_code/400'), 
    # ('internlm3_8b_chat-long_rush_250308_32k_code-2000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250308_32k_code/2000'), 
    # ('internlm3_8b_chat-long_rush_250308_32k_code-4000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250308_32k_code/4000'), 

    # ('internlm3_8b_chat-long_rush_250309_32k_code-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250309_32k_code/400/'), 
    # ('internlm3_8b_chat-long_rush_250309_32k_code-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250309_32k_code/1000/'), 

    # ('internlm3_8b_chat-long_rush_250310_128k_lwj1-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250310_128k_code/400/'), 
    # ('internlm3_8b_chat-long_rush_250312_32k_cpt-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250312_32k_cpt/1000'), 
    # ('internlm3_8b_chat-long_rush_250313_32k_cpt-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250313_32k_cpt/1000'), 
    # ('internlm3_8b_chat-long_rush_250313a_32k_cpt-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_rush-internlm3_8b-250313a_32k_cpt/1000/'), 

    # ('llama3_2_3b_chat-long_safety_250203-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_2_3b-250203/400/'), 
    # ('llama3_2_3b_chat-long_safety_250203-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_2_3b-250203/1000/'), 

    # ('llama3_1_8b_chat-long_safety_250203-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250203/400/'), 
    # ('llama3_1_8b_chat-long_safety_250203-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-llama3_1_8b-250203/1000/'), 

    # ('qwen2_5_1b_chat-long_safety_250203-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_1b-250203/400/'), 
    # ('qwen2_5_1b_chat-long_safety_250203-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_1b-250203/1000/'), 

    # ('qwen2_5_7b_chat-long_safety_250203-400-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_7b-250203/400/'), 
    # ('qwen2_5_7b_chat-long_safety_250203-1000-32k', '/cpfs01/shared/llm_ddd/liuxiaoran/tmp_ckpts_hf/long_safe-qwen2_5_7b-250203/1000/'), 

]

models = [
    dict(
        type=HuggingFaceBaseModel, abbr=abbr, path=path, 
        tokenizer_path=meta_dict[abbr.split('-')[0]] if abbr.split('-')[0] in meta_dict else path, 
        model_kwargs={'attn_implementation': 'flash_attention_2', }, 
        max_seq_len=31500, max_out_len=500, batch_size=1, run_cfg=dict(num_gpus=1),
    ) for abbr, path in models
]

work_dir = './outputs_xrliu/long_safety_long/'

alillm_ha_h2_cfg = dict(
    bashrc_path="/cpfs01/user/liuxiaoran/.bashrc",
    conda_env_name="/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1",
    python_env_path="/cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1",
    dlc_config_path="/cpfs01/shared/llm_ddd/liuxiaoran/dlc.cfg",
    workspace_id="ws1h2vgufjufr4jj",  # 'ws1f6e9s7dh69ttt',  # "ws1ujefpjyfgqjwp",  # 
    worker_image="pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/paieflops/tanghuanze:haijun-fastchat-zk-1022",
    resource_id="",
    data_sources=["data1bgvj0n14to0", "data14t59tzld8y5", "data14gz11jajrrt", "datayfnabr11a497", ], 
    dlc_job_cmd="create",
    priority=4,
    enable_preemptible_job=True
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=alillm_ha_h2_cfg,
        max_num_workers=64, retry=2, 
        task=dict(type=OpenICLInferTask),
    ),
)

# eval = dict(
#     partitioner=dict(type=NaivePartitioner, n=1),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=8,
#         task=dict(type=OpenICLEvalTask, dump_details=True), 
#     ),
# )

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=alillm_ha_h2_cfg,
        max_num_workers=16, retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# source /cpfs01/user/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.1
# python run.py eval_xrliu/eval_xrliu_long_safety_long.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_long_safety_long.py --dump-eval-details -r 20240820_190019 第一次用
