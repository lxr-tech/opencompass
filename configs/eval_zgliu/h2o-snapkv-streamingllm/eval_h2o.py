from mmengine.config import read_base

with read_base():
    # from .datasets.obqa.obqa_ppl import obqa_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

    # from opencompass.configs.datasets.ruler.ruler_32k_gen import niah_datasets

datasets = []
datasets += ARC_e_datasets
datasets += ARC_c_datasets

from opencompass.models import Llama2_H2O
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
        type=Llama2_H2O,
        path='/remote-home1/share/models/llama3_hf/Meta-Llama-3-8B-Instruct/',
        model_kwargs=dict(device_map='auto', trust_remote_code = True, torch_dtype = torch.bfloat16),
        tokenizer_path='/remote-home1/share/models/llama3_hf/Meta-Llama-3-8B-Instruct/',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code = True),
        max_seq_len=32*1024,
        max_out_len=50,
        batch_size=1,
        # run_cfg=dict(num_gpus=2),
    )
]
