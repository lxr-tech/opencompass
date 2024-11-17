from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
import torch

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k

datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
datasets += ruler_datasets_32k

from opencompass.models import HuggingFaceCausalLM
models = [
        dict(
            abbr="llama3.1-8b",
            type=HuggingFaceCausalLM,
            path='/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
            tokenizer_path='/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=32*1024,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
        ),
        dict(
            abbr="llama3.1-8b-instruct",
            type=HuggingFaceCausalLM,
            path='/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16,
                              rope_scaling={"type": "dynamic", "factor": 4.0}),
            tokenizer_path='/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=32*1024,
            max_out_len=50, 
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
        )
    ]

work_dir = './outputs/ruler_eval_llama3.1/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=3,  # 最大并行运行进程数
        task=dict(type=OpenICLInferTask),  # 待运行的任务
    )
)