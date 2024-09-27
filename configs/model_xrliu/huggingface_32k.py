
from opencompass.models import HuggingFaceModel

import torch
import numpy as np

path_dict = {'llama2_7B': '', 
             'llama2_7B_chat': '', 
             'llama2_13B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55/', 
             'llama3_8B': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B/', 
             'llama3_8B_chat': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B-Instruct/', 
             'llama3_1_8B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/', 
             'llama3_1_8B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/', 
             'llama3_1_70B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/7740ff69081bd553f4879f71eebcc2d6df2fbcb3/', 
             'internlm2_1B': '',
             'internlm2_1B_base': '',
             'internlm2_7B': '',
             'internlm2_7B_base': '',  
             'internlm2_7B_chat': '',
             'internlm2_20B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2-20b/snapshots/f689c06034d1bcaa47f611f428ec9dd9bcc92789/',
             'internlm2.5_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b/snapshots/7bee5c5d7c4591d7f081f5976610195fdd1f1e35/',  # '/nas/shared/alillm2/zhangshuo/exported_transformers/official_Ampere2.5_7B_Enhance_2.0.0/50000/', 
             'internlm2.5_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b-chat-1m/snapshots/8d1a709a04d71440ef3df6ebbe204672f411c8b6/', 
             'internlm2.5_20B': '/cpfs02/puyu/shared/alillm2/alillm2/zhangshuo/exported_transformers/official_InternLM2-5_20B_Enhance_18-0-0_256k_FixBBHLeak_wsd_from_50000_to_52500_5-0-0_256k/52600/', 

             'qwen1_5_14B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen1.5-14B/snapshots/dce4b190d34470818e5bec2a92cb8233aaa02ca2/', 
             'qwen1_5_32B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen1.5-32B/snapshots/cefef80dc06a65f89d1d71d0adbc56d335ca2490/', 
             'qwen2_1B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a/', 
             'qwen2_1B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8/', 
             'qwen2_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B/snapshots/d8412313bc00677839b4e38cdc751d536ccc12ab/', 
             'qwen2_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B-Instruct/snapshots/6ddb532fa75db9a1269de86eaa579818eb39743a/', 
             'qwen2_72B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-72B/snapshots/e5cebedf0244946b1ad5eb2d753ca3c1a90f11fb/', 
             'mistral3_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8/', 
             'mistral3_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de/', 
             'glm4_9B_chat_1M': '/nas/shared/public/llmeval/model_weights/hf_hub/models--THUDM--glm-4-9b-chat-1m/snapshots/715ddbe91082f976ff6a4ca06d59e5bbff6c3642/', 

             'moss2_13B_v2_1000': '/nas/shared/public/liuxiaoran/tmp_ckpts_hf/moss2-13b-v2/1000/', 
             'moss2_13B_v2_4000': '/nas/shared/public/liuxiaoran/tmp_ckpts_hf/moss2-13b-v2/4000/', 
            }

num_gpus = {'llama2_7B': 1, 'llama2_7B_chat': 1, 'llama2_13B': 2, 
            'llama3_8B': 1, 'llama3_8B_chat': 1, 'llama3_1_8B': 1, 'llama3_1_8B_chat': 1, 'llama3_1_70B': 4, 
            'qwen1_5_14B': 8, 'qwen1_5_32B': 8, 
            'qwen2_1B': 1, 'qwen2_1B_chat': 1, 'qwen2_7B': 1, 'qwen2_7B_chat': 1, 'qwen2_72B': 4, 
            'internlm2_7B': 1, 'internlm2_7B_chat': 1, 
            'internlm2.5_7B': 1, 'internlm2.5_7B_chat': 1, 
            'internlm2_1B': 1, 'internlm2_20B': 2, 'internlm2.5_20B': 2, 
            'mistral3_7B': 1, 'mistral3_7B_chat': 1, 
            'glm4_9B_chat_1M': 1, 

            'moss2_13B_v2_1000': 2, 'moss2_13B_v2_4000': 2, 
            }


tags = [
        # ('-fp16-32k_cat', 'internlm2.5_7B', 'internlm2.5_7B', '{prompt}', 31500, 
        # {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        # ('-bf16-32k_cat', 'internlm2.5_7B', 'internlm2.5_7B', '{prompt}', 31500, 
        # {'torch_dtype': 'bfloat16', 'device_map': 'auto',  'trust_remote_code': True}), 
        # ('-fp16-local-32k_cat', 'internlm2.5_7B', 'internlm2.5_7B', '{prompt}', 31500, 
        # {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True, 'local_files_only': True}), 
        # ('-bf16-local-32k_cat', 'internlm2.5_7B', 'internlm2.5_7B', '{prompt}', 31500, 
        # {'torch_dtype': 'bfloat16', 'device_map': 'auto',  'trust_remote_code': True, 'local_files_only': True}), 

        # ('-32k_cat', 'moss2_13B_v2_1000', 'moss2_13B_v2_1000', '{prompt}', 31500, 
        # {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ('-32k_cat', 'moss2_13B_v2_4000', 'moss2_13B_v2_4000', '{prompt}', 31500, 
        {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 

        ('-32k_cat', 'qwen1_5_14B', 'qwen1_5_14B', '{prompt}', 31500, 
        {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ('-32k_cat', 'qwen1_5_32B', 'qwen1_5_32B', '{prompt}', 31500, 
        {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ('-32k_cat', 'internlm2_20B', 'internlm2_20B', '{prompt}', 31500, 
        {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ('-32k_cat', 'internlm2.5_20B', 'internlm2.5_20B', '{prompt}', 31500, 
        {'torch_dtype': 'float16', 'device_map': 'auto',  'trust_remote_code': True}), 
        ('-4k_cat', 'llama2_13B', 'llama2_13B', '{prompt}', 3500, 
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

