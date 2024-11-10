import torch
from mmengine.config import read_base

with read_base():
    # from opencompass.configs.datasets.collections.base_medium_llama import piqa_datasets, siqa_datasets
    # from opencompass.configs.datasets.mmlu.mmlu_all_sets import mmlu_all_sets
    # from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    # from opencompass.configs.datasets.ceval.cev
    # from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from opencompass.configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets
    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets

datasets = []
# datasets += hellaswag_datasets
# datasets += winogrande_datasets
# datasets += truthfulqa_datasets
# datasets += truthfulqa_datasets
# datasets += mmlu_all_sets
# datasets += ceval_datasets
datasets += gsm8k_datasets
datasets += ARC_e_datasets
datasets += ARC_c_datasets
datasets += hellaswag_datasets
datasets += winogrande_datasets
datasets += truthfulqa_datasets


from opencompass.models import HuggingFaceCausalLM
from opencompass.models import LightCache_MossHuaweiForCausalLM
models = [
        # 原始 huaweimoss
        # dict(
        #     type=HuggingFaceCausalLM,
        #     path='/remote-home1/zgliu/models/huawei_model',
        #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
        #     tokenizer_path='/remote-home1/zgliu/models/huawei_model/',
        #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        #     max_seq_len=2048,
        #     max_out_len=50,
        #     run_cfg=dict(num_gpus=1, num_procs=1),
        #     batch_size=128,
        # ),

        # Lightcache 结合的 huaweimoss
        dict(
            type=LightCache_MossHuaweiForCausalLM,
            path='/remote-home1/zgliu/models/huawei_model',
            long_cache_config_kwargs=dict(
                global_size=4, mid_size=4, span_size=32, 
                local_size=2048, chunk_size=1024, 
                rope_scaling=None, recall_option='default', 
                unique_option='group_unique', recall_clip=64,
                ),
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
            tokenizer_path='/remote-home1/zgliu/models/huawei_model/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=2048,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=128,
        ),

        # Baichuan13b-base
        # dict(
        #     type=HuggingFaceCausalLM,
        #     path='/remote-home1/zgliu/models/Baichuan-13B-Base/',
        #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
        #     tokenizer_path='/remote-home1/zgliu/models/Baichuan-13B-Base/',
        #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        #     max_seq_len=2048,
        #     max_out_len=50,
        #     run_cfg=dict(num_gpus=1, num_procs=1),
        #     batch_size=128,
        # ),
    ]
