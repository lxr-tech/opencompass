from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPCausalLM_v0827

with read_base():

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
    
    from opencompass.configs.datasets.leval.levaltpo.leval_tpo_gen import LEval_tpo_datasets
    from opencompass.configs.datasets.leval.levalgsm100.leval_gsm100_gen import LEval_gsm100_datasets
    from opencompass.configs.datasets.leval.levalquality.leval_quality_gen import LEval_quality_datasets
    from opencompass.configs.datasets.leval.levalcoursera.leval_coursera_gen import LEval_coursera_datasets
    from opencompass.configs.datasets.leval.levaltopicretrieval.leval_topic_retrieval_gen import LEval_tr_datasets
    from opencompass.configs.datasets.leval.levalscientificqa.leval_scientificqa_gen import LEval_scientificqa_datasets
    
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

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-vanilla-dclm-90000decay-lctx': 'rope-0904-376m-4k-vanilla-ckpt90000-decay-lctx', 
    '376m-imagh-dclm-90000decay-lctx': 'rope-0904-376m-4k-imagh-ckpt90000-decay-lctx', 
    '376m-imago-dclm-90000decay-lctx': 'rope-0904-376m-4k-imago-ckpt90000-decay-lctx', 

    # 776m
    '776m-vanilla-fw100b': 'rope-0906-776m-4k-vanilla', 

    '776m-vanilla-dclm-90000decay': 'rope-0904-776m-4k-vanilla-ckpt90000-decay', 
    '776m-imagh-dclm-90000decay': 'rope-0904-776m-4k-imagh-ckpt90000-decay', 
    '776m-imago-dclm-90000decay': 'rope-0904-776m-4k-imago-ckpt90000-decay', 

    '776m-vanilla-dclm-90000decay-lctx': 'rope-0904-776m-4k-vanilla-ckpt90000-decay-lctx', 
    '776m-imagh-dclm-90000decay-lctx': 'rope-0904-776m-4k-imagh-ckpt90000-decay-lctx', 
    '776m-imago-dclm-90000decay-lctx': 'rope-0904-776m-4k-imago-ckpt90000-decay-lctx', 

    '776m-vanilla-dclm-130428decay': 'rope-0904-776m-4k-vanilla-ckpt130428-decay', 
    '776m-imagh-dclm-130428decay': 'rope-0904-776m-4k-imagh-ckpt130428-decay', 
    '776m-imago-dclm-130428decay': 'rope-0904-776m-4k-imago-ckpt130428-decay', 

    '776m-vanilla-dclm': 'rope-0904-776m-4k-vanilla', 
    '776m-imagh-dclm': 'rope-0904-776m-4k-imagh', 
    '776m-imago-dclm': 'rope-0904-776m-4k-imago', 
    '776m-path-dclm': 'rope-0904-776m-4k-path', 
    '776m-fope-dclm': 'rope-0904-776m-4k-fope', 

    '776m-vanilla-fw2-cpt': 'rope-0901-776m-4k-vanilla-ckpt36231-cpt', 
    '776m-imagh-fw2-cpt': 'rope-0901-776m-4k-imagh-ckpt36231-cpt', 
    '776m-imago-fw2-cpt': 'rope-0901-776m-4k-imago-ckpt36231-cpt', 

    '776m-vanilla-fw2': 'rope-0901-776m-4k-vanilla', 
    '776m-imagh-fw2': 'rope-0901-776m-4k-imagh', 
    '776m-imago-fw2': 'rope-0901-776m-4k-imago', 

    '776m-vanilla-fw': 'rope-0831-776m-4k-vanilla', 
    '776m-imagh-fw': 'rope-0831-776m-4k-imagh', 
    '776m-imago-fw': 'rope-0831-776m-4k-imago', 

    # '776m-vanilla': 'rope-0820-776m-4k-vanilla', 
    # '776m-imagh': 'rope-0827-776m-4k-imagh', 
    # '776m-imago': 'rope-0827-776m-4k-imago', 

    # '776m-1dl': 'rope-0827-776m-4k-1dl', 
    # '776m-1d2': 'rope-0820-776m-4k-1d2', 

    # '776m-imag2': 'rope-0820-776m-4k-imag2', 
    # '776m-imag1': 'rope-0820-776m-4k-imag1', 

}

models = [

    ('376m-vanilla-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 31500, 500), 
    ('376m-imagh-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 6000, 31500, 500), 
    ('376m-imago-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 6000, 31500, 500), 

    ('776m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 8000, 31500, 500), 
    ('776m-imagh-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 8000, 31500, 500), 
    ('776m-imago-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 8000, 31500, 500), 

    # # ('776m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 3500, 500), 
    # # ('776m-imagh-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10001, 3500, 500), 
    # # ('776m-imago-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10001, 3500, 500), 

    # # ('776m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 3500, 500), 
    # # ('776m-imagh-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10001, 3500, 500), 
    # # ('776m-imago-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10001, 3500, 500), 

    # # ('776m-vanilla-dclm-90000decay-ckpt-ntk3-8k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 7500, 500), 
    # # ('776m-imagh-dclm-90000decay-ckpt-ntk3-8k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 10001, 7500, 500), 
    # # ('776m-imago-dclm-90000decay-ckpt-ntk3-8k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 10001, 7500, 500), 

    # # ('776m-vanilla-dclm-90000decay-lctx-ckpt-32k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 31500, 500), 
    # # ('776m-imagh-dclm-90000decay-lctx-ckpt-32k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10000, 31500, 500), 
    # # ('776m-imago-dclm-90000decay-lctx-ckpt-32k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10000, 31500, 500), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt-16k', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 15500, 500), 

    # # ('776m-vanilla-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 14493, 3500, 500), 
    # # ('776m-imagh-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 14493, 3500, 500), 
    # # ('776m-imago-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 14493, 3500, 500), 

]

models = [
    dict(
        type=RoPEPPCausalLM_v0827, abbr=abbr, rope_config=rope_config, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        model_kwargs={'flash_attention': True}, max_seq_len=max_seq_len, max_out_len=max_out_len,  # max_out_len=64 before 0607
        drop_middle=True, batch_size=1, run_cfg=dict(num_gpus=1, num_procs=1),
    ) for abbr, rope_config, ckpt, max_seq_len, max_out_len in models
]

work_dir = './outputs_xrliu/rope_pp_long-v0827/'

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
# conda activate llm-fepe
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_ruler.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_ruler.py --dump-eval-details -r 20240820_190019 第一次用
