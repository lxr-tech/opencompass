from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
import torch

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets

datasets = []
datasets += ruler_datasets

from opencompass.models import HuggingFaceCausalLM
models = [
        dict(
            abbr="llama2-7b",
            type=HuggingFaceCausalLM,
            path='/remote-home1/share/models/llama_v2_hf/7b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
                            #   rope_scaling={"type": "dynamic", "factor": 4.0}),
            tokenizer_path='/remote-home1/share/models/llama_v2_hf/7b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=32*1024,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=16,
        ),
        dict(
            abbr="llama3-7b-chat",
            type=HuggingFaceCausalLM,
            path='/remote-home1/share/models/llama_v2_hf/7b-chat/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16,),
                            #   rope_scaling={"type": "dynamic", "factor": 4.0}),
            tokenizer_path='/remote-home1/share/models/llama_v2_hf/7b-chat/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=32*1024,
            max_out_len=50, 
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=2,
        )
    ]

work_dir = './outputs/ruler_eval_llama2/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=3,  # 最大并行运行进程数
        task=dict(type=OpenICLInferTask),  # 待运行的任务
    )
)