from mmengine.config import read_base

with read_base():

    from ..datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # modified in 240907

    from ..datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets                             # zero shot
    from ..datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets                             # zero shot

    from ..datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets                                # 5-shot
    from ..datasets.ceval.ceval_ppl_578f8d import ceval_datasets                             # 5-shot
    from ..datasets.cmmlu.cmmlu_ppl_8b9c76 import cmmlu_datasets
    
    from ..datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from ..datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from ..datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from ..datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    from ..datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from ..datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from ..datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from ..datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    from ..datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from ..datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from ..datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from ..datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    from ..datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from ..datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets
    from ..datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from ..datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets

    from ..datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from ..datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from ..datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    from ..datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from ..datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
