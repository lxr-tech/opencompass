from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
import torch

with read_base():
    # from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets

datasets = []
# datasets += ruler_datasets
datasets += ARC_c_datasets
datasets += ARC_e_datasets

# from opencompass.models import HuggingFaceCausalLM
from opencompass.models.zgliu.huggingface_for_long import HuggingFaceModelForLong
from opencompass.models.zgliu.huggingface_full_nearby import HuggingFaceModelFullNearby

models = [
        dict(
            abbr="llama3-8b-nearby",
            type=HuggingFaceModelFullNearby,
            path='/remote-home1/zgliu/models/llama3-8b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
                            #   rope_scaling={"type": "dynamic", "factor": 4.0}),
            tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=32*1024,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
        ),
        # dict(
        #     abbr="llama3-8b",
        #     type=HuggingFaceModelForLong,
        #     path='/remote-home1/zgliu/models/llama3-8b/',
        #     model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16),
        #                     #   rope_scaling={"type": "dynamic", "factor": 4.0}),
        #     tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
        #     tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        #     max_seq_len=32*1024,
        #     max_out_len=50,
        #     run_cfg=dict(num_gpus=1, num_procs=1),
        #     batch_size=1,
        # ),
    ]

work_dir = './outputs/ruler_eval_llama3-nearby/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=3,  # 最大并行运行进程数
        task=dict(type=OpenICLInferTask),  # 待运行的任务
    )
)