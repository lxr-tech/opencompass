from opencompass.models.zgliu.huggingface_for_long import HuggingFaceModelForLong
from opencompass.models.zgliu.light_cache.light_cache_wrapper import LightCacheCausalLM 

import torch
import numpy as np

# def generate_integer_list(start, end, length=32):
#     """
#     生成在两个指定整数之间的均匀整数列表，列表长度为指定的长度。

#     参数：
#         start (int): 列表起始值。
#         end (int): 列表结束值（包含该值）。
#         length (int): 列表的长度。

#     返回：
#         list: 一个包含从 start 到 end 之间均匀分布的整数列表。
#     """
#     if length <= 0:
#         raise ValueError("列表长度必须大于0")
    
#     if start == end:
#         return [start] * length
    
#     step = (end - start) / (length - 1)
#     return [round(start + i * step) for i in range(length)]

path_dict = {
             'llama3_8B': '/remote-home1/zgliu/models/llama3-8b', 
            }

tags = [
        ############## standard   
        # (
        #     LightCacheCausalLM, 'clip127-ntk4-pe-original-true', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,},
        # ),
        # (
        #     LightCacheCausalLM, 'clip127-ntk4-pe-original-false', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': False, 'q_cache_len': 0, 
        #     'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,},
        # ),
        ###############
        
        ##########################
        # 11.12测试
        (
            LightCacheCausalLM, 'keyfill-pe-quant-4first', 'llama3_8B', 'llama', '{prompt}', -1,
            dict(device_map='cuda', torch_dtype=torch.float16),
            {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
            'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
            'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
            'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
            'use_fill': [True]*4+[False]*28,},
        ),
        # (
        #     LightCacheCausalLM, 'keyfill-pe-quant-16first', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [True]*16+[False]*16,},
        # ),
        # (
        #     LightCacheCausalLM, 'valuefill-4first', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [True]*4 + [False]*28,},
        # ),
        # (
        #     LightCacheCausalLM, 'valuefill-16first', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [True]*16 + [False]*16,},
        # ),
        
        ###########################
        # 11.11测试
        # (
        #     LightCacheCausalLM, 'full-attn', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_full_attn': [True]*32,},
        # ),
        # (
        #     LightCacheCausalLM, 'full-4first', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_full_attn': [True]*4 + [False]*28,},
        # ),
        # (
        #     LightCacheCausalLM, 'full-16first', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_full_attn': [True]*16 + [False]*16,},
        # ),
        # (
        #     LightCacheCausalLM, 'full-28first', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_full_attn': [True]*28 + [False]*4,},
        # ),
        # (
        #     LightCacheCausalLM, 'full-16last', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 4}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_full_attn': [False]*16 + [True]*16,},
        # ),

        #### 11.10 test
        # (
        #     LightCacheCausalLM, 'clip127-ntk4-fill-first2', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [True]*2+[False]*30,}
        # ),
        # (
        #     LightCacheCausalLM, 'clip127-ntk4', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     # 'use_fill': [True]*2+[False]*30,
        #     }
        # ),
        
        # (
        #     HuggingFaceModelForLong, 'hf-long', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16, attn_implementation='flash_attention_2', rope_scaling={"type": "dynamic", "factor": 4.0}),
        #     None,
        # ),
        # (
        #     LightCacheCausalLM, 'clip127-ntk4-fill-first1', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [True]*1+[False]*31,}
        # ),
        # (
        #     LightCacheCausalLM, 'fill-first16', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [True]*16+[False]*16,}
        # ),
        # (
        #     LightCacheCausalLM, 'fill-last16', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [False]*16+[True]*16,}
        # ),
        # (
        #     LightCacheCausalLM, 'fill-last8', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [False]*24+[True]*8,}
        # ),
        # (
        #     LightCacheCausalLM, 'fill-last2', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [False]*28+[True]*4,}
        # ),
        # (
        #     LightCacheCausalLM, 'fill-mid16', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [False]*8+[True]*16+[False]*8}
        # ),
        # (
        #     LightCacheCausalLM, 'fill-mid8', 'llama3_8B', 'llama', '{prompt}', -1,
        #     dict(device_map='cuda', torch_dtype=torch.float16),
        #     {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 1024, 
        #     'first_prefill': 8192, 'recall_type': 'qk_pe', 'pe_original': True, 'q_cache_len': 0, 
        #     'rope_scaling': {"type": "scaling", "factor": 2}, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #     'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 127, 'max_position_embeddings': 8192,
        #     'use_fill': [False]*12+[True]*8+[False]*12}
        # ),
    ]

models = []

for model_class, abbr, group, model_type, prompt_format, long_bench_cat, model_kwargs, long_cache_config in tags:
    abbr += '-' + group
    model_dict = {'attn_implementation': 'flash_attention_2'}
    model_dict = {}
    model_dict.update(
        dict(
            type=model_class, 
            abbr=f'{group}{abbr}',
            path=path_dict[group],
            model_type=model_type,
            tokenizer_path=path_dict[group],
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', 
                                  use_fast=False, trust_remote_code=True), 
            max_out_len=50,
            max_seq_len=32*1024,
            batch_size=1,
            model_kwargs=model_kwargs,
            long_cache_config=long_cache_config,
            long_bench_cat=long_bench_cat,
            prompt_format=prompt_format, 
            batch_padding=False, # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=1, 
                        #  num_procs=1,  # num_gpus[group.split('-')[0]]
                         ), 
            # quanto_enable=abbr.__contains__('quanto'), 
            # chat_enable=False,  # group.__contains__('chat'),             
        )
    )

    models.append(model_dict)
