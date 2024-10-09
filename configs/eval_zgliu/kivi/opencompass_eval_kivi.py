import torch
from mmengine.config import read_base

with read_base():
    # from opencompass.configs.datasets.collections.base_medium_llama import piqa_datasets, siqa_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from opencompass.configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets
    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets
    # from opencompass.configs.datasets.mmlu.mmlu_all_sets import mmlu_all_sets
    # from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    # from opencompass.configs.datasets.ceval.cev
    # from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

datasets = []
# datasets += hellaswag_datasets
# datasets += winogrande_datasets
# datasets += truthfulqa_datasets
# datasets += truthfulqa_datasets
# datasets += mmlu_all_sets
# datasets += ceval_datasets
datasets += ARC_c_datasets
datasets += ARC_e_datasets

from opencompass.models import HuggingFaceCausalLM
from opencompass.models.kivi_model import KIVI_LlamaForCausalLM
models = [
        # dict(
        #     type=KIVI_LlamaForCausalLM,
        #     path='/remote-home1/zgliu/models/llama2-7b',
        #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        #     tokenizer_path='/remote-home1/zgliu/models/llama2-7b',
        #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        #     max_seq_len=2048,
        #     max_out_len=50,
        #     run_cfg=dict(num_gpus=1, num_procs=1),
        #     batch_size=64,
        # ),
        # dict(
        #     type=HuggingFaceCausalLM,
        #     path='/remote-home1/zgliu/models/llama2-7b',
        #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
        #     tokenizer_path='/remote-home1/zgliu/models/llama2-7b',
        #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        #     max_seq_len=2048,
        #     max_out_len=50,
        #     run_cfg=dict(num_gpus=1, num_procs=1),
        #     batch_size=64,
        # ),
        dict(
            type=KIVI_LlamaForCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            tokenizer_path='/remote-home1/zgliu/models/llama2-7b',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=2048,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=64,
        ),
        dict(
            type=HuggingFaceCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            tokenizer_path='/remote-home1/zgliu/models/llama2-7b',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=2048,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=64,
        ),
    ]
