import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
import torch.nn.functional as F

# from transformers import LlamaForCausalLM
from transformers import AutoConfig
from transformers import AutoTokenizer
import datetime
# import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# from .flash_utils_v2.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
# from .flash_utils_v2.AttnCache import AttnCacheConfig

# from collie import CollieConfig

PromptType = Union[PromptList, str]

from ..huggingface import HuggingFaceCausalLM, BaseModel
from transformers.generation.utils import GenerationConfig

def print_gpu_memory_info():
    # 获取当前设备的显存信息
    current_device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    allocated_memory = torch.cuda.memory_allocated(current_device)
    cached_memory = torch.cuda.memory_cached(current_device)
    
    # 转换为GB单位
    total_memory_gb = total_memory / (1024 ** 3)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    cached_memory_gb = cached_memory / (1024 ** 3)
    
    # 计算使用百分比
    usage_percentage = (allocated_memory / total_memory) * 100
    
    print('*'*100)
    print(f"Device: {current_device}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB ({usage_percentage:.2f}% used)")
    print(f"Cached Memory: {cached_memory_gb:.2f} GB")
    print('*'*100)

@MODELS.register_module()
class LightCacheCausalLM(HuggingFaceCausalLM):
    def __init__(self,
                 path: str,
                 model_type: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 long_cache_config = None,
                 long_bench_cat = -1,
                 prompt_format: str = '{prompt}',
                 attn_implementation: str = 'eager', 
                 quanto_enable: bool = False, 
                 chat_enable: bool = False):
        BaseModel.__init__(self, path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode

        self.long_cache_config = long_cache_config
        self.attn_implementation = attn_implementation
        self.model_type = model_type

        self._load_tokenizer(path=path, 
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_type=model_type, 
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

        self.long_bench_cat = long_bench_cat
        self.prompt_format = prompt_format
                
        self.quanto_enable = quanto_enable
        self.chat_enable = chat_enable
        
        self.past_key_values = None
        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=1, do_sample=False, use_cache=True
        )


    def _load_tokenizer(self, path: Optional[str], tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)  # , local_files_only=True

        super()._load_tokenizer(path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs)
        
        # if 'llama2' in tokenizer_path.lower() or 'llama-2' in tokenizer_path.lower():
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.eos_token_id
            # self.tokenizer.unk_token_id = self.tokenizer.bos_token_id

        # self.eos_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.bos_token = '<s>'
        # if self.model_type in ['llama', 'internlm2', ]: 
        #     self.tokenizer.eos_token = '</s>'
        #     self.pad_token_id = self.tokenizer.pad_token_id
        # else:
        #     self.pad_token_id = self.tokenizer.eos_token_id

    def _load_model(self,
                    path: str, 
                    model_type: str, 
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        
        # config = CollieConfig.from_pretrained(path, trust_remote_code=True)  # , local_files_only=True

        # config.model_config.use_cache = True
        # config.model_config.attn_implementation = self.attn_implementation
        # config.checkpointing = True
        # config.use_flash = True
        # config.ds_config = {
        #     'bf16': {
        #         'enabled': True,
        #     },
        #     'train_micro_batch_size_per_gpu': 1, 
        #     'zero_optimization': {
        #         "stage": 3, 
        #     },            
        # }
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.config.attn_implementation = self.attn_implementation
        # self.long_cache_config['score_record'] = True
                
        if model_type == 'llama':
            from .light_cache_llama import LlamaForCausalLM
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            # print(self.long_cache_config, flush=True)
            # if self.long_cache_config.get('cpu_offload', False):
            from .cache_utils import LightCacheConfig
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
                                                num_attention_heads=self.config.num_attention_heads, 
                                                **self.long_cache_config)
            # else:
            # from .cache_utils_offload import LightCacheConfig
            # self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
            #                                     num_attention_heads=self.config.num_attention_heads, 
            #                                     **self.long_cache_config)
            # print('use cpu_offload !', flush=True)
            self.config.long_cache_config = self.long_cache_config
            self.config.rope_scaling = self.long_cache_config.rope_scaling
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
                                                          config=self.config, trust_remote_code=True,  # local_files_only=True, 
                                                          attn_implementation=self.attn_implementation)
        elif model_type == 'llama+':

            # name_list = ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.k_proj.weight', 'model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.o_proj.weight', 'model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.up_proj.weight', 'model.layers.0.mlp.down_proj.weight', 'model.layers.0.input_layernorm.weight', 'model.layers.0.post_attention_layernorm.weight', 'model.layers.1.self_attn.q_proj.weight', 'model.layers.1.self_attn.k_proj.weight', 'model.layers.1.self_attn.v_proj.weight', 'model.layers.1.self_attn.o_proj.weight', 'model.layers.1.mlp.gate_proj.weight', 'model.layers.1.mlp.up_proj.weight', 'model.layers.1.mlp.down_proj.weight', 'model.layers.1.input_layernorm.weight', 'model.layers.1.post_attention_layernorm.weight', 'model.layers.2.self_attn.q_proj.weight', 'model.layers.2.self_attn.k_proj.weight', 'model.layers.2.self_attn.v_proj.weight', 'model.layers.2.self_attn.o_proj.weight', 'model.layers.2.mlp.gate_proj.weight', 'model.layers.2.mlp.up_proj.weight', 'model.layers.2.mlp.down_proj.weight', 'model.layers.2.input_layernorm.weight', 'model.layers.2.post_attention_layernorm.weight', 'model.layers.3.self_attn.q_proj.weight', 'model.layers.3.self_attn.k_proj.weight', 'model.layers.3.self_attn.v_proj.weight', 'model.layers.3.self_attn.o_proj.weight', 'model.layers.3.mlp.gate_proj.weight', 'model.layers.3.mlp.up_proj.weight', 'model.layers.3.mlp.down_proj.weight', 'model.layers.3.input_layernorm.weight', 'model.layers.3.post_attention_layernorm.weight', 'model.layers.4.self_attn.q_proj.weight', 'model.layers.4.self_attn.k_proj.weight', 'model.layers.4.self_attn.v_proj.weight', 'model.layers.4.self_attn.o_proj.weight', 'model.layers.4.mlp.gate_proj.weight', 'model.layers.4.mlp.up_proj.weight', 'model.layers.4.mlp.down_proj.weight', 'model.layers.4.input_layernorm.weight', 'model.layers.4.post_attention_layernorm.weight', 'model.layers.5.self_attn.q_proj.weight', 'model.layers.5.self_attn.k_proj.weight', 'model.layers.5.self_attn.v_proj.weight', 'model.layers.5.self_attn.o_proj.weight', 'model.layers.5.mlp.gate_proj.weight', 'model.layers.5.mlp.up_proj.weight', 'model.layers.5.mlp.down_proj.weight', 'model.layers.5.input_layernorm.weight', 'model.layers.5.post_attention_layernorm.weight', 'model.layers.6.self_attn.q_proj.weight', 'model.layers.6.self_attn.k_proj.weight', 'model.layers.6.self_attn.v_proj.weight', 'model.layers.6.self_attn.o_proj.weight', 'model.layers.6.mlp.gate_proj.weight', 'model.layers.6.mlp.up_proj.weight', 'model.layers.6.mlp.down_proj.weight', 'model.layers.6.input_layernorm.weight', 'model.layers.6.post_attention_layernorm.weight', 'model.layers.7.self_attn.q_proj.weight', 'model.layers.7.self_attn.k_proj.weight', 'model.layers.7.self_attn.v_proj.weight', 'model.layers.7.self_attn.o_proj.weight', 'model.layers.7.mlp.gate_proj.weight', 'model.layers.7.mlp.up_proj.weight', 'model.layers.7.mlp.down_proj.weight', 'model.layers.7.input_layernorm.weight', 'model.layers.7.post_attention_layernorm.weight', 'model.layers.8.self_attn.q_proj.weight', 'model.layers.8.self_attn.k_proj.weight', 'model.layers.8.self_attn.v_proj.weight', 'model.layers.8.self_attn.o_proj.weight', 'model.layers.8.mlp.gate_proj.weight', 'model.layers.8.mlp.up_proj.weight', 'model.layers.8.mlp.down_proj.weight', 'model.layers.8.input_layernorm.weight', 'model.layers.8.post_attention_layernorm.weight', 'model.layers.9.self_attn.q_proj.weight', 'model.layers.9.self_attn.k_proj.weight', 'model.layers.9.self_attn.v_proj.weight', 'model.layers.9.self_attn.o_proj.weight', 'model.layers.9.mlp.gate_proj.weight', 'model.layers.9.mlp.up_proj.weight', 'model.layers.9.mlp.down_proj.weight', 'model.layers.9.input_layernorm.weight', 'model.layers.9.post_attention_layernorm.weight', 'model.layers.10.self_attn.q_proj.weight', 'model.layers.10.self_attn.k_proj.weight', 'model.layers.10.self_attn.v_proj.weight', 'model.layers.10.self_attn.o_proj.weight', 'model.layers.10.mlp.gate_proj.weight', 'model.layers.10.mlp.up_proj.weight', 'model.layers.10.mlp.down_proj.weight', 'model.layers.10.input_layernorm.weight', 'model.layers.10.post_attention_layernorm.weight', 'model.layers.11.self_attn.q_proj.weight', 'model.layers.11.self_attn.k_proj.weight', 'model.layers.11.self_attn.v_proj.weight', 'model.layers.11.self_attn.o_proj.weight', 'model.layers.11.mlp.gate_proj.weight', 'model.layers.11.mlp.up_proj.weight', 'model.layers.11.mlp.down_proj.weight', 'model.layers.11.input_layernorm.weight', 'model.layers.11.post_attention_layernorm.weight', 'model.layers.12.self_attn.q_proj.weight', 'model.layers.12.self_attn.k_proj.weight', 'model.layers.12.self_attn.v_proj.weight', 'model.layers.12.self_attn.o_proj.weight', 'model.layers.12.mlp.gate_proj.weight', 'model.layers.12.mlp.up_proj.weight', 'model.layers.12.mlp.down_proj.weight', 'model.layers.12.input_layernorm.weight', 'model.layers.12.post_attention_layernorm.weight', 'model.layers.13.self_attn.q_proj.weight', 'model.layers.13.self_attn.k_proj.weight', 'model.layers.13.self_attn.v_proj.weight', 'model.layers.13.self_attn.o_proj.weight', 'model.layers.13.mlp.gate_proj.weight', 'model.layers.13.mlp.up_proj.weight', 'model.layers.13.mlp.down_proj.weight', 'model.layers.13.input_layernorm.weight', 'model.layers.13.post_attention_layernorm.weight', 'model.layers.14.self_attn.q_proj.weight', 'model.layers.14.self_attn.k_proj.weight', 'model.layers.14.self_attn.v_proj.weight', 'model.layers.14.self_attn.o_proj.weight', 'model.layers.14.mlp.gate_proj.weight', 'model.layers.14.mlp.up_proj.weight', 'model.layers.14.mlp.down_proj.weight', 'model.layers.14.input_layernorm.weight', 'model.layers.14.post_attention_layernorm.weight', 'model.layers.15.self_attn.q_proj.weight', 'model.layers.15.self_attn.k_proj.weight', 'model.layers.15.self_attn.v_proj.weight', 'model.layers.15.self_attn.o_proj.weight', 'model.layers.15.mlp.gate_proj.weight', 'model.layers.15.mlp.up_proj.weight', 'model.layers.15.mlp.down_proj.weight', 'model.layers.15.input_layernorm.weight', 'model.layers.15.post_attention_layernorm.weight', 'model.layers.16.self_attn.q_proj.weight', 'model.layers.16.self_attn.k_proj.weight', 'model.layers.16.self_attn.v_proj.weight', 'model.layers.16.self_attn.o_proj.weight', 'model.layers.16.mlp.gate_proj.weight', 'model.layers.16.mlp.up_proj.weight', 'model.layers.16.mlp.down_proj.weight', 'model.layers.16.input_layernorm.weight', 'model.layers.16.post_attention_layernorm.weight', 'model.layers.17.self_attn.q_proj.weight', 'model.layers.17.self_attn.k_proj.weight', 'model.layers.17.self_attn.v_proj.weight', 'model.layers.17.self_attn.o_proj.weight', 'model.layers.17.mlp.gate_proj.weight', 'model.layers.17.mlp.up_proj.weight', 'model.layers.17.mlp.down_proj.weight', 'model.layers.17.input_layernorm.weight', 'model.layers.17.post_attention_layernorm.weight', 'model.layers.18.self_attn.q_proj.weight', 'model.layers.18.self_attn.k_proj.weight', 'model.layers.18.self_attn.v_proj.weight', 'model.layers.18.self_attn.o_proj.weight', 'model.layers.18.mlp.gate_proj.weight', 'model.layers.18.mlp.up_proj.weight', 'model.layers.18.mlp.down_proj.weight', 'model.layers.18.input_layernorm.weight', 'model.layers.18.post_attention_layernorm.weight', 'model.layers.19.self_attn.q_proj.weight', 'model.layers.19.self_attn.k_proj.weight', 'model.layers.19.self_attn.v_proj.weight', 'model.layers.19.self_attn.o_proj.weight', 'model.layers.19.mlp.gate_proj.weight', 'model.layers.19.mlp.up_proj.weight', 'model.layers.19.mlp.down_proj.weight', 'model.layers.19.input_layernorm.weight', 'model.layers.19.post_attention_layernorm.weight', 'model.layers.20.self_attn.q_proj.weight', 'model.layers.20.self_attn.k_proj.weight', 'model.layers.20.self_attn.v_proj.weight', 'model.layers.20.self_attn.o_proj.weight', 'model.layers.20.mlp.gate_proj.weight', 'model.layers.20.mlp.up_proj.weight', 'model.layers.20.mlp.down_proj.weight', 'model.layers.20.input_layernorm.weight', 'model.layers.20.post_attention_layernorm.weight', 'model.layers.21.self_attn.q_proj.weight', 'model.layers.21.self_attn.k_proj.weight', 'model.layers.21.self_attn.v_proj.weight', 'model.layers.21.self_attn.o_proj.weight', 'model.layers.21.mlp.gate_proj.weight', 'model.layers.21.mlp.up_proj.weight', 'model.layers.21.mlp.down_proj.weight', 'model.layers.21.input_layernorm.weight', 'model.layers.21.post_attention_layernorm.weight', 'model.layers.22.self_attn.q_proj.weight', 'model.layers.22.self_attn.k_proj.weight', 'model.layers.22.self_attn.v_proj.weight', 'model.layers.22.self_attn.o_proj.weight', 'model.layers.22.mlp.gate_proj.weight', 'model.layers.22.mlp.up_proj.weight', 'model.layers.22.mlp.down_proj.weight', 'model.layers.22.input_layernorm.weight', 'model.layers.22.post_attention_layernorm.weight', 'model.layers.23.self_attn.q_proj.weight', 'model.layers.23.self_attn.k_proj.weight', 'model.layers.23.self_attn.v_proj.weight', 'model.layers.23.self_attn.o_proj.weight', 'model.layers.23.mlp.gate_proj.weight', 'model.layers.23.mlp.up_proj.weight', 'model.layers.23.mlp.down_proj.weight', 'model.layers.23.input_layernorm.weight', 'model.layers.23.post_attention_layernorm.weight', 'model.layers.24.self_attn.q_proj.weight', 'model.layers.24.self_attn.k_proj.weight', 'model.layers.24.self_attn.v_proj.weight', 'model.layers.24.self_attn.o_proj.weight', 'model.layers.24.mlp.gate_proj.weight', 'model.layers.24.mlp.up_proj.weight', 'model.layers.24.mlp.down_proj.weight', 'model.layers.24.input_layernorm.weight', 'model.layers.24.post_attention_layernorm.weight', 'model.layers.25.self_attn.q_proj.weight', 'model.layers.25.self_attn.k_proj.weight', 'model.layers.25.self_attn.v_proj.weight', 'model.layers.25.self_attn.o_proj.weight', 'model.layers.25.mlp.gate_proj.weight', 'model.layers.25.mlp.up_proj.weight', 'model.layers.25.mlp.down_proj.weight', 'model.layers.25.input_layernorm.weight', 'model.layers.25.post_attention_layernorm.weight', 'model.layers.26.self_attn.q_proj.weight', 'model.layers.26.self_attn.k_proj.weight', 'model.layers.26.self_attn.v_proj.weight', 'model.layers.26.self_attn.o_proj.weight', 'model.layers.26.mlp.gate_proj.weight', 'model.layers.26.mlp.up_proj.weight', 'model.layers.26.mlp.down_proj.weight', 'model.layers.26.input_layernorm.weight', 'model.layers.26.post_attention_layernorm.weight', 'model.layers.27.self_attn.q_proj.weight', 'model.layers.27.self_attn.k_proj.weight', 'model.layers.27.self_attn.v_proj.weight', 'model.layers.27.self_attn.o_proj.weight', 'model.layers.27.mlp.gate_proj.weight', 'model.layers.27.mlp.up_proj.weight', 'model.layers.27.mlp.down_proj.weight', 'model.layers.27.input_layernorm.weight', 'model.layers.27.post_attention_layernorm.weight', 'model.layers.28.self_attn.q_proj.weight', 'model.layers.28.self_attn.k_proj.weight', 'model.layers.28.self_attn.v_proj.weight', 'model.layers.28.self_attn.o_proj.weight', 'model.layers.28.mlp.gate_proj.weight', 'model.layers.28.mlp.up_proj.weight', 'model.layers.28.mlp.down_proj.weight', 'model.layers.28.input_layernorm.weight', 'model.layers.28.post_attention_layernorm.weight', 'model.layers.29.self_attn.q_proj.weight', 'model.layers.29.self_attn.k_proj.weight', 'model.layers.29.self_attn.v_proj.weight', 'model.layers.29.self_attn.o_proj.weight', 'model.layers.29.mlp.gate_proj.weight', 'model.layers.29.mlp.up_proj.weight', 'model.layers.29.mlp.down_proj.weight', 'model.layers.29.input_layernorm.weight', 'model.layers.29.post_attention_layernorm.weight', 'model.layers.30.self_attn.q_proj.weight', 'model.layers.30.self_attn.k_proj.weight', 'model.layers.30.self_attn.v_proj.weight', 'model.layers.30.self_attn.o_proj.weight', 'model.layers.30.mlp.gate_proj.weight', 'model.layers.30.mlp.up_proj.weight', 'model.layers.30.mlp.down_proj.weight', 'model.layers.30.input_layernorm.weight', 'model.layers.30.post_attention_layernorm.weight', 'model.layers.31.self_attn.q_proj.weight', 'model.layers.31.self_attn.k_proj.weight', 'model.layers.31.self_attn.v_proj.weight', 'model.layers.31.self_attn.o_proj.weight', 'model.layers.31.mlp.gate_proj.weight', 'model.layers.31.mlp.up_proj.weight', 'model.layers.31.mlp.down_proj.weight', 'model.layers.31.input_layernorm.weight', 'model.layers.31.post_attention_layernorm.weight', 'model.norm.weight', 'lm_head.weight']
            # name_dict = {'model.embed_tokens.weight': 0, 'model.norm.weight': 1, 'lm_head.weight': 1, }
            # for i in range(0, 2):
            #     name_dict[f'model.layers.{i}'] = 0
            # for i in range(2, 32):
            #     name_dict[f'model.layers.{i}'] = 1
            # name_dict = {}

            # for name in name_list[:10] + name_list[-11:]:
            #     name_dict[name] = 0
            #     # if '.0.' in name:
            #     # elif '.1.' in name:
            #     #     name_dict[name] = 1
            #     # elif '.2.' in name:
            #     #     name_dict[name] = 1
            # for name in name_list[10:-11]:
            #     name_dict[name] = 1

            from .cache_utils import LightCacheConfig
            from .light_cache_llama3_1 import LlamaForCausalLM
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
                                                 num_attention_heads=self.config.num_attention_heads, 
                                                 **self.long_cache_config)
            self.config.long_cache_config = self.long_cache_config
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # name_dict,  # device,  # **model_kwargs,
                                                          config=self.config, trust_remote_code=True,  # local_files_only=True, 
                                                          attn_implementation=self.attn_implementation)
            # print(self.model.hf_device_map)

        elif model_type == 'internlm2':
            from .cache_utils import LightCacheConfig
            from .light_cache_internlm2 import InternLM2ForCausalLM
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
                                                 num_attention_heads=self.config.num_attention_heads, 
                                                 **self.long_cache_config)
            self.config.long_cache_config = self.long_cache_config
            self.config.rope_scaling = self.long_cache_config.rope_scaling
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            # self.config = config.model_config
            
            # from collie import setup_distribution
            # setup_distribution(config=config)
            # import torch.distributed as dist
            # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

            # # Set the device
            # device = torch.device(f"cuda:{int(os.environ['RANK'])}")
            
            self.model = InternLM2ForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
                                                              config=self.config, trust_remote_code=True,  # local_files_only=True, 
                                                              attn_implementation=self.attn_implementation)
            # # self.model.eval().cuda()
            
            # from torch.optim import AdamW
            # self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
            # self.model.eval()
            
        elif model_type == 'qwen2':
            from .cache_utils import LightCacheConfig
            from .light_cache_qwen2 import Qwen2ForCausalLM
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
                                                 num_attention_heads=self.config.num_attention_heads, 
                                                 **self.long_cache_config)
            self.config.long_cache_config = self.long_cache_config
            self.config.rope_scaling = self.long_cache_config.rope_scaling
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
                                                          config=self.config, trust_remote_code=True,  # local_files_only=True, 
                                                          attn_implementation=self.attn_implementation)

        elif model_type == 'mistral':
            from .cache_utils import LightCacheConfig
            from .light_cache_mistral import MistralForCausalLM
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_key_value_heads, 
                                                 num_attention_heads=self.config.num_attention_heads, 
                                                 **self.long_cache_config)
            self.config.long_cache_config = self.long_cache_config
            self.config.rope_scaling = self.long_cache_config.rope_scaling
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = MistralForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
                                                          config=self.config, trust_remote_code=True,  # local_files_only=True, 
                                                          attn_implementation=self.attn_implementation)

        elif model_type == 'chatglm':
            from .cache_utils import LightCacheConfig
            from .light_cache_chatglm import ChatGLMForConditionalGeneration
            # assert type(self.long_cache_config) == dict, f"long_cache_config must be a dict, but got {type(self.long_cache_config)}"
            self.long_cache_config = LightCacheConfig(num_key_value_heads=self.config.num_attention_heads // self.config.multi_query_group_num, 
                                                 num_attention_heads=self.config.num_attention_heads, 
                                                 **self.long_cache_config)
            self.config._attn_implementation = "flash_attention_2"
            self.config.long_cache_config = self.long_cache_config
            self.config.rope_scaling = self.long_cache_config.rope_scaling
            print('self.config.rope_scaling', self.config.rope_scaling, flush=True)
            self.config.pretraining_tp = 1
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = ChatGLMForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.float16, device_map="auto",  # device,  # **model_kwargs,
                config=self.config, trust_remote_code=True)

        else:
            # self.config = config.model_config
            self.long_cache_config = {
                'global_size': 32, 'mid_size': 1, 'span_size': 32, 'local_size': 4096, 'chunk_size': 512, 
                'rope_scaling': None, 'recall_option': 'full_attn', 'unique_option': 'group_unique', 
                'key_compress_ratio': 1, 'value_compress_ratio': 1, }
            from .cache_utils import LightCacheConfig
            self.long_cache_config = LightCacheConfig(**self.long_cache_config)
            self.config.long_cache_config = self.long_cache_config
            from transformers import AutoModelForCausalLM
            self.config._flash_attn_2_enabled = True
            self.config.attn_implementation = "flash_attention_2"
            self.config._attn_implementation = "flash_attention_2"
            self._set_model_kwargs_torch_dtype(model_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs, config=self.config)
        
        # import ipdb
        # ipdb.set_trace()
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]: 
        """Generate results given a list of inputs."""
        self.model.eval()  # Set the model to evaluation mode
        outputs_text = []
        
        # import sys
        
        # sys.path.append('/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/light_cache/')
        
        # import threading
        # from profiles import CUDAMemoryProfiler
                    
        # memory_profiler = CUDAMemoryProfiler(
        #     [self.model],
        #     filename='/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/light_cache/profile2.txt'
        # )

        # sys.settrace(memory_profiler)
        # threading.settrace(memory_profiler)   

        # from torch.profiler import profile, record_function, ProfilerActivity

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
             
        for text in inputs:
            
            if text.endswith('\n\n') and not self.chat_enable:
                text = text[:-2]
            
            if self.chat_enable:
                input_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], 
                                                            tokenize=True, add_generation_prompt=True, 
                                                            return_tensors="pt")
            else:
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids

            if self.long_bench_cat > 0:
                if input_ids.shape[-1] > self.long_bench_cat:
                    input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
            
            print(f"\n\ninput_ids.shape: {input_ids.shape}\n", flush=True)
            
            if self.long_cache_config.recall_option not in ['generate_only', 'full_attn', ]:
                seq_len = input_ids.shape[-1]
                mod_len = (seq_len - 1 - self.long_cache_config.global_size - self.long_cache_config.local_size) % 128
                local_size = self.long_cache_config.local_size - (128 - mod_len)
                start, end = 0, self.long_cache_config.global_size + local_size  # + self.long_cache_config.mid_size 
                chunk_length = self.long_cache_config.chunk_size
                # print(f'{type(self.past_key_values)}, {self.past_key_values is None}', flush=True)
                while start < seq_len - 1:
                    if start // 100000 != end // 100000:
                        print(datetime.datetime.now(), start, end, flush=True)
                    input_chunk = input_ids[:, start:min(end, seq_len - 1)]
                    outputs = self.model.forward(input_chunk, past_key_values=self.past_key_values)
                    start, end = end, end + chunk_length
                    self.past_key_values = outputs.past_key_values
                    torch.cuda.empty_cache()
                outputs = self.model.generate(input_ids, past_key_values=self.past_key_values, return_dict_in_generate=True, 
                                            max_new_tokens=max_out_len, generation_config=self.generation_config)
            else:
                outputs = self.model.generate(input_ids, return_dict_in_generate=True, 
                                            max_new_tokens=max_out_len, generation_config=self.generation_config)
                
            self.past_key_values = outputs.past_key_values
            self.past_key_values = self.past_key_values.clean_cache()
            generated_text = self.tokenizer.decode(outputs.sequences[0, input_ids.shape[1]:], skip_special_tokens=True)

            outputs_text.append(generated_text)
            torch.cuda.empty_cache()
        
        # print(prof.table())
        # prof.export_chrome_trace('/fs-computility/llm/shared/liuxiaoran/opencompass/opencompass/models/light_cache/resnet_profile2.json')

        return outputs_text