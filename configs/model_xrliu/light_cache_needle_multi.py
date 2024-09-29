
from opencompass.models.xrliu.light_cache.light_cache_wrapper import LightCacheCausalLM

import torch
import numpy as np

path_dict = {'llama2_7B': '', 
             'llama2_7B_chat': '', 
             'llama2_13B': '', 
             'llama3_8B': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B/', 
             'llama3_8B_chat': '/cpfs01/shared/alillm2/llm_llama3_hf/Meta-Llama-3-8B-Instruct/', 
             'llama3_1_8B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/', 
             'llama3_1_8B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/', 
             'llama3_1_70B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3.1-70B/snapshots/7740ff69081bd553f4879f71eebcc2d6df2fbcb3/', 
             'llama3_2_3B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72/', 
             'llama3_2_1B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14/', 
             'internlm2_1B': '',
             'internlm2_1B_base': '',
             'internlm2_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2-7b/snapshots/fa732f7cd8ec6299400e37e790ecd82abb1a7f9a/',
             'internlm2_7B_base': '',  
             'internlm2_7B_chat': '',
             'internlm2_20B': '',
             'internlm2.5_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b/snapshots/7bee5c5d7c4591d7f081f5976610195fdd1f1e35/',  # '/nas/shared/alillm2/zhangshuo/exported_transformers/official_Ampere2.5_7B_Enhance_2.0.0/50000/', 
             'internlm2.5_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--internlm--internlm2_5-7b-chat-1m/snapshots/8d1a709a04d71440ef3df6ebbe204672f411c8b6/', 
             'qwen2_1B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a/', 
             'qwen2_1B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8/', 
             'qwen2_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B/snapshots/d8412313bc00677839b4e38cdc751d536ccc12ab/', 
             'qwen2_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-7B-Instruct/snapshots/6ddb532fa75db9a1269de86eaa579818eb39743a/', 
             'qwen2_72B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--Qwen--Qwen2-72B/snapshots/e5cebedf0244946b1ad5eb2d753ca3c1a90f11fb/', 
             'mistral3_7B': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8/', 
             'mistral3_7B_chat': '/nas/shared/public/llmeval/model_weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de/', 
             'glm4_9B_chat_1M': '/nas/shared/public/llmeval/model_weights/hf_hub/models--THUDM--glm-4-9b-chat-1m/snapshots/715ddbe91082f976ff6a4ca06d59e5bbff6c3642/', 
            }

num_gpus = {'llama2_7B': 4, 'llama2_7B_chat': 4, 'llama2_13B': 8, 
            'llama3_8B': 4, 'llama3_8B_chat': 4, 'llama3_1_8B': 4, 'llama3_1_8B_chat': 4, 'llama3_1_70B': 8, 
            'qwen1.5_1B': 4, 'qwen1.5_7B': 4, 'qwen1.5_14B': 8, 'qwen1.5_32B': 8, 
            'qwen2_1B': 4, 'qwen2_1B_chat': 4, 'qwen2_7B': 4, 'qwen2_7B_chat': 4, 'qwen2_72B': 8, 
            'internlm2_7B': 4, 'internlm2_7B_chat': 4, 
            'internlm2.5_7B': 4, 'internlm2.5_7B_chat': 4, 
            'internlm2_1B': 4, 'internlm2_20B': 4, 
            'mistral3_7B': 4, 'mistral3_7B_chat': 4, 
            'glm4_9B_chat_1M': 4, 
            }

tags = [
        ('-32.4x32.8192.2048-clip256-qk', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1,  
         {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
          'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
          'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
          'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }),
        ('-32.4x32.8192.2048-clip256-qk', 'llama3_2_3B_chat', 'llama+', '{prompt}', -1,  
         {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
          'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
          'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
          'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }),
        ('-32.4x32.8192.2048-clip256-qk', 'mistral3_7B_chat', 'mistral', '{prompt}', -1,  
         {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
          'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
          'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
          'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }),
        ('-32.4x32.8192.2048-clip256-qk', 'internlm2.5_7B_chat', 'internlm2', '{prompt}', -1,  
         {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
          'first_prefill': 8192, 'recall_type': 'qk', 'pe_original': False, 'q_cache_len': 0, 
          'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
          'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }),

        # ('', 'llama3_8B', 'llama', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-ntk_d4', 'llama3_8B', 'llama', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': {'type': 'dynamic', 'factor': 4, }, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.4096.512', 'llama3_8B', 'llama', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.4x32.4096.512-hu-clip96', 'llama3_8B', 'llama', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 96, }),  
        
        # ('', 'llama3_1_8B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'llama3_1_8B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'llama3_1_8B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.2048-hu-clip1024', 'llama3_1_8B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 1024, }), 
        # ('-32.1x32.8192.1024-hu-clip512', 'llama3_1_8B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        # ('-32.1x32.8192.512-hu-clip256', 'llama3_1_8B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

        # ('-32.4x32.2048.512-tl-clip191', 'llama3_8B_chat', 'llama', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 2048, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 191, }), 

        # ('', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.4x32.8192.512-tl-clip256', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-tl-clip256', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.2048-tl-clip256', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.512-tl-clip256', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.2048-tl-clip256', 'llama3_1_8B_chat', 'llama+', '{prompt}', -1, 
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 2048, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        
        # ('', 'llama3_1_70B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'llama3_1_70B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'llama3_1_70B', 'llama+', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        
        # ('', 'internlm2_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'internlm2_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'internlm2_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        
        # ('', 'internlm2_1B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'internlm2_1B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'internlm2_1B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        
        # ('', 'internlm2.5_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'internlm2.5_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'internlm2.5_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.1024-hu-clip512', 'internlm2.5_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 

        # ('-32.1x32.8192.1024-hu-clip512', 'internlm2_7B', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        # ('-32.1x32.8192.1024-hu-clip512', 'internlm2.5_7B_chat', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        # ('-32.4x32.8192.1024-hu-clip512', 'internlm2.5_7B_chat', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        # ('-32.8x32.8192.1024-hu-clip512', 'internlm2.5_7B_chat', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 8, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        # ('-32.lnx32.8192.1024-hu-clip512', 'internlm2.5_7B_chat', 'internlm2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': -1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        
        # ('', 'qwen2_1B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'qwen2_1B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'qwen2_1B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.1024-hu-clip512', 'qwen2_1B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 

        # ('-32.1x32.8192.1024-hu-clip512', 'qwen2_1B_chat', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 

        # ('', 'qwen2_7B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'qwen2_7B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'qwen2_7B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.1024-hu-clip512', 'qwen2_7B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        
        # ('', 'qwen2_72B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'qwen2_72B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'qwen2_72B', 'qwen2', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

        # ('', 'mistral3_7B', 'mistral', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'mistral3_7B', 'mistral', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'mistral3_7B', 'mistral', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 

        # ('', 'glm4_9B_chat_1M', 'chatglm', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 64, }),
        # ('-32.0.8192.512', 'glm4_9B_chat_1M', 'chatglm', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 0, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.4x32.8192.512-hu-clip256', 'glm4_9B_chat_1M', 'chatglm', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 4, 'span_size': 32, 'local_size': 8192, 'chunk_size': 512, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 256, }), 
        # ('-32.1x32.8192.1024-hu-clip512', 'glm4_9B_chat_1M', 'chatglm', '{prompt}', -1,  
        #  {'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 8192, 'chunk_size': 1024, 
        #   'rope_scaling': None, 'recall_option': 'default', 'unique_option': 'head_unique', 
        #   'key_compress_ratio': 1, 'value_compress_ratio': 1, 'recall_clip': 512, }), 
        
        ]

models = []

for abbr, group, model_type, prompt_format, long_bench_cat, long_cache_config in tags:

    model_dict = {'attn_implementation': 'flash_attention_2'}
    model_dict.update(
        dict(
            type=LightCacheCausalLM, 
            abbr=f'{group}{abbr}',
            path=path_dict[group],
            model_type=model_type,
            tokenizer_path=path_dict[group],
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', 
                                  use_fast=False, trust_remote_code=True), 
            max_out_len=250,  # 50,  # 
            # max_seq_len=4096,
            batch_size=1,
            model_kwargs=dict(device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True),
            long_cache_config=long_cache_config,
            long_bench_cat=long_bench_cat,
            prompt_format=prompt_format, 
            batch_padding=False, # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=1,  # num_gpus[group.split('-')[0]], #  
                         num_procs=1,  # num_gpus[group.split('-')[0]]
                         ), 
            quanto_enable=abbr.__contains__('quanto'), 
            chat_enable=group.__contains__('chat'),  # change it back in 240908    
        )
    )

    models.append(model_dict)
