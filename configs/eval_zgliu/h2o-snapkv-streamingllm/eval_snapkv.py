from mmengine.config import read_base

with read_base():
    # from .datasets.obqa.obqa_ppl import obqa_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

datasets = [*ARC_e_datasets, *ARC_c_datasets]

from opencompass.models import Llama2_SnapKV
from opencompass.models import HuggingFaceBaseModel
import torch

models = [
    # dict(
    #     type=HuggingFaceBaseModel,
    #     abbr='llama-2-7b-hf',
    #     path='/remote-home/yrsong/models/llama2-7b/Llama-2-7b-hf',
    #     max_out_len=1024,
    #     batch_size=8,
    #     run_cfg=dict(num_gpus=1),
    # ), 
    dict(
        type=Llama2_SnapKV,
        path='/remote-home1/yrsong/models/llama2-7b/Llama-2-7b-hf',
        model_kwargs=dict(device_map='auto', trust_remote_code = True, torch_dtype = torch.bfloat16),
        tokenizer_path='/remote-home1/yrsong/models/llama2-7b/Llama-2-7b-hf',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code = True),
        max_seq_len=2048,
        max_out_len=50,
        batch_size=64,
        # run_cfg=dict(num_gpus=2),
    )
]
