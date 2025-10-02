from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import PaTHCausalLM

with read_base():

    from opencompass.configs.datasets.babilong.babilong_0k_gen import babiLong_0k_datasets
    from opencompass.configs.datasets.babilong.babilong_2k_gen import babiLong_2k_datasets
    from opencompass.configs.datasets.babilong.babilong_4k_gen import babiLong_4k_datasets
    from opencompass.configs.datasets.babilong.babilong_8k_gen import babiLong_8k_datasets
    from opencompass.configs.datasets.babilong.babilong_16k_gen import babiLong_16k_datasets
    from opencompass.configs.datasets.babilong.babilong_32k_gen import babiLong_32k_datasets
    from opencompass.configs.datasets.babilong.babilong_64k_gen import babiLong_64k_datasets
    from opencompass.configs.datasets.babilong.babilong_128k_gen import babiLong_128k_datasets

    from opencompass.configs.datasets.lveval.lveval import LVEval_datasets

datasets = []

# datasets += LVEval_datasets

datasets += babiLong_0k_datasets
datasets += babiLong_2k_datasets
datasets += babiLong_4k_datasets
datasets += babiLong_8k_datasets
datasets += babiLong_16k_datasets
# datasets += babiLong_32k_datasets
# datasets += babiLong_64k_datasets
# datasets += babiLong_128k_datasets

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {
    '776m-path-dclm': 'rope-0904-776m-4k-path', 

    '376m-path-dclm': 'rope-0904-376m-4k-path', 
}

models = [
    #### 776m

    # ('776m-path-dclm-ckpt10000', None, 10000, 64), 
    # ('776m-path-dclm-ckpt20000', None, 20000, 64), 
    # ('776m-path-dclm-ckpt30000', None, 30000, 64), 
    # ('776m-path-dclm-ckpt40000', None, 40000, 64), 

    # ('776m-path-dclm-ckpt50000', None, 50000, 64), 
    # ('776m-path-dclm-ckpt60000', None, 60000, 64), 
    # ('776m-path-dclm-ckpt80000', None, 80000, 64), 
    # ('776m-path-dclm-ckpt100000', None, 100000, 64), 

    #### 376m

    ('376m-path-dclm-ckpt10000', None, 10000, 64), 
    ('376m-path-dclm-ckpt20000', None, 20000, 64), 
    ('376m-path-dclm-ckpt30000', None, 30000, 64), 
    ('376m-path-dclm-ckpt40000', None, 40000, 64), 

    ('376m-path-dclm-ckpt50000', None, 50000, 64), 
    # ('376m-path-dclm-ckpt60000', None, 60000, 64), 
    # ('376m-path-dclm-ckpt80000', None, 80000, 64), 
    # ('376m-path-dclm-ckpt100000', None, 100000, 64), 
]

models = [
    dict(
        type=PaTHCausalLM, abbr=abbr, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, rope_config, ckpt, max_out_len in models
]

work_dir = './outputs_xrliu/rope_pp_babilong-v0827/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),  # dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# summarizer = dict(
#     dataset_abbrs=['ruler_4k', 'ruler_8k', 'ruler_16k', 'ruler_32k', 'ruler_128k'],
#     summary_groups=sum(
#         [v for k, v in locals().items() if k.endswith('_summary_groups')], []
#     ),
# )
 
# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate llm-fepe
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_ruler.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_ruler.py --dump-eval-details -r 20240820_190019 第一次用
