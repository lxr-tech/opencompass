from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
import torch

with read_base():
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

    from opencompass.configs.datasets.ruler.ruler_32k_gen import niah_datasets

datasets = []
datasets += niah_datasets
datasets += ARC_e_datasets
datasets += ARC_c_datasets

from opencompass.models import HuggingFaceCausalLM
from opencompass.models import ChunkLlama_LlamaForCausalLM

models = [
        # dict(
        #     abbr="llama3-8b-instruct-ntk4",
        #     type=HuggingFaceModelForLong,
        #     path='/remote-home1/share/models/llama3_hf/Meta-Llama-3-8B-Instruct/',
        #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16,
        #                       attn_implementation="flash_attention_2",
        #                       rope_scaling={"type": "dynamic", "factor": 4.0}),
        #     tokenizer_path='/remote-home1/share/models/llama3_hf/Meta-Llama-3-8B-Instruct/',
        #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        #     max_seq_len=32*1024,
        #     max_out_len=50, 
        #     run_cfg=dict(num_gpus=1, num_procs=1),
        #     batch_size=1,
        # ),
        dict(
            abbr="llama3-8b-base-chunkllama",
            type=ChunkLlama_LlamaForCausalLM,
            path='/remote-home1/share/models/llama3_2_hf/Llama-3.2-3B',
            model_kwargs=dict(trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2",
                              device_map="cuda"),
            tokenizer_path='/remote-home1/share/models/llama3_2_hf/Llama-3.2-3B',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=32*1024,
            max_out_len=50, 
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
        ),
    ]

work_dir = './outputs/ttt/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=3,  # 最大并行运行进程数, 不指定时为 -1
        task=dict(type=OpenICLInferTask),  # 待运行的任务
    )
)