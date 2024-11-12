from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
import torch

with read_base():
    # from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets
    from opencompass.configs.datasets.ruler.ruler_32k_gen_sample_multikey import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_8k_gen_sample_multikey import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_4k_gen_sample_multikey import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_16k_gen_sample_multikey import ruler_datasets

    from .model_llama3_reAttn import models
    # from opencompass.configs.datasets.ruler.ruler_32k_gen

datasets = []
datasets += ruler_datasets

# from opencompass.models import HuggingFaceCausalLM
from opencompass.models.zgliu.huggingface_for_long import HuggingFaceModelForLong
from opencompass.models.zgliu.light_cache.light_cache_wrapper import LightCacheCausalLM

# models = [
#         dict(
#             abbr="llama3-8b-instruct-ntk4",
#             type=HuggingFaceModelForLong,
#             path='/remote-home1/zgliu/models/llama3-8b',
#             model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16,
#                               attn_implementation="flash_attention_2",
#                               rope_scaling={"type": "dynamic", "factor": 4.0}),
#             tokenizer_path='/remote-home1/zgliu/models/llama3-8b',
#             tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
#             max_seq_len=32*1024,
#             max_out_len=50, 
#             run_cfg=dict(num_gpus=1, num_procs=1),
#             batch_size=1,
#         ),
#         dict(
#             abbr="llama3-8b-instruct-lightcache-default",
#             type=LightCacheCausalLM,
#             path='/remote-home1/share/models/llama3_hf/Meta-Llama-3-8B-Instruct/',
#             model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16,
#                               attn_implementation="flash_attention_2",
#                               rope_scaling={"type": "dynamic", "factor": 4.0}),
#             long_cache_config=long_cache_config,
#             tokenizer_path='/remote-home1/share/models/llama3_hf/Meta-Llama-3-8B-Instruct/',
#             tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
#             max_seq_len=32*1024,
#             max_out_len=50, 
#             run_cfg=dict(num_gpus=1, num_procs=1),
#             batch_size=1,
#         ),
#     ]



work_dir = './outputs/ruler_eval_reAttn/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=1,  # 最大并行运行进程数
        task=dict(type=OpenICLInferTask),  # 待运行的任务
    )
)