
from opencompass.models import HuggingFaceModel

import torch
import numpy as np

path_dict = {
    'internlm2_5_7B': '/cpfs01/shared/public/pretrain_transfer_data/puyu_transfer_data/guohonglin/hf_hub/models--internlm--internlm2_5-7b/snapshots/daa886c96bc86f54f03c725db5316adbbc3eb5db/',  # '/nas/shared/alillm2/zhangshuo/exported_transformers/official_Ampere2.5_7B_Enhance_2.0.0/50000/', 
    'internlm2_5_7B-lctx_delivery_32k_241210-3000': '/cpfs01/shared/public/liuxiaoran/tmp_ckpts_hf/lctx_delivery-internlm2_5_7b-241209/3000/', 
    'internlm2_5_7B-lctx_delivery_32k_241210-1000': '/cpfs01/shared/public/liuxiaoran/tmp_ckpts_hf/lctx_delivery-internlm2_5_7b-241209/1000/', 
}

num_gpus = {'llama2_7B': 1, 'llama2_7B_chat': 1, 'llama2_13B': 2, 
            'llama3_8B': 1, 'llama3_8B_chat': 1, 'llama3_1_8B': 1, 'llama3_1_8B_chat': 1, 'llama3_1_70B': 4, 
            'llama3_2_3B': 1,  
            'qwen1_5_14B': 8, 'qwen1_5_32B': 8, 
            'qwen2_1B': 1, 'qwen2_1B_chat': 1, 'qwen2_7B': 1, 'qwen2_7B_chat': 1, 'qwen2_72B': 4, 
            'internlm2_7B': 1, 'internlm2_7B_chat': 1, 
            'internlm2_5_7B': 1, 'internlm2.5_7B_chat': 1, 
            'internlm2_1B': 1, 'internlm2_20B': 2, 'internlm2.5_20B': 2, 
            'mistral3_7B': 1, 'mistral3_7B_chat': 1, 
            'glm4_9B_chat_1M': 1, 

            'moss2_13B_v2_1000': 2, 'moss2_13B_v2_4000': 2, 

            'yxg_24101101a': 1, 'Ampere2_5_7B_base': 1, 'Ampere2_5_7B_enhance': 1, 
            }


tags = [
        ('-32k_cat', 'internlm2_5_7B', 'internlm2_5_7B', '{prompt}', 31500, 
         {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ('-32k_cat', 'internlm2_5_7B-lctx_delivery_32k_241210-3000', 'internlm2_5_7B-lctx_delivery_32k_241210-3000', '{prompt}', 31500, 
         {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ]


models = [
    dict(
        abbr='{}{}'.format(group, abbr),  # name in path
        type=HuggingFaceModel, 
        model_kwargs=model_kwargs,
        model_type=group.split('_')[0],  # ['llama2', 'internlm2', 'chatglm3']
        model_path=path_dict[path],
        config_path=path_dict[group.split('-')[0]],
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                              trust_remote_code=True, use_fast=False, ),
        max_out_len=128,  # no use
        max_seq_len=32768,
        long_bench_cat=cat_len, 
        prompt_format=prompt_format, 
        batch_size=1, 
        batch_padding=True,
        run_cfg=dict(num_gpus=num_gpus[group.split('-')[0]], 
                     num_procs=1),  # tp or pp size
    ) for abbr, group, path, prompt_format, cat_len, model_kwargs in tags]

