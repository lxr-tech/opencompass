from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import FoPECausalLM

with read_base():
    # from opencompass.configs.datasets.ruler.ruler_2k_gen import ruler_datasets as ruler_datasets_2k
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k

datasets = []
# datasets += ruler_datasets_2k
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
# datasets += ruler_datasets_32k

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-fope-dclm-v2': 'rope-0918-376m-4k-fope', 
    '376m-path-dclm': 'rope-0904-376m-4k-path', 
    
    '376m-fope-dclm-v2-90000decay': 'rope-0918-376m-4k-fope-ckpt90000-decay', 

    '776m-fope-dclm-v2': 'rope-0918-776m-4k-fope', 
    '776m-path-dclm': 'rope-0904-776m-4k-path', 

    '776m-fope-dclm-v2-90000decay': 'rope-0918-776m-4k-fope-ckpt90000-decay', 

}

models = [

    ('376m-fope-dclm-v2-ckpt10000', None, 10000, 64), 
    ('376m-path-dclm-ckpt10000', None, 10000, 64), 

    ('376m-fope-dclm-v2-ckpt20000', None, 20000, 64), 
    ('376m-path-dclm-ckpt20000', None, 20000, 64), 

    ('376m-fope-dclm-v2-ckpt30000', None, 30000, 64), 
    ('376m-path-dclm-ckpt30000', None, 30000, 64), 
    
    ('376m-fope-dclm-v2-ckpt40000', None, 40000, 64), 
    ('376m-path-dclm-ckpt40000', None, 40000, 64), 
    
    ('376m-fope-dclm-v2-ckpt50000', None, 50000, 64), 
    ('376m-path-dclm-ckpt50000', None, 50000, 64), 

    # ('376m-fope-dclm-v2-ckpt60000', None, 60000, 64), 
    # ('376m-fope-dclm-v2-ckpt80000', None, 80000, 64), 
    # ('376m-fope-dclm-v2-ckpt100000', None, 100000, 64), 
    # ('376m-fope-dclm-v2-ckpt100000-ntk3', {'scaling_factor': 3}, 100000, 64), 
    # ('376m-fope-dclm-v2-ckpt120000', None, 120000, 64), 

    # ('376m-fope-dclm-v2-40000decay-ckpt5000', None, 5000, 64), 
    # ('376m-fope-dclm-v2-40000decay', None, 10000, 64), 
    # ('376m-fope-dclm-v2-90000decay', None, 10000, 64), 
    # ('376m-fope-dclm-v2-90000decay-ckpt-ntk3', {'scaling_factor': 3}, 10001, 64), 

    # ('776m-fope-dclm-v2-ckpt10000', None, 10000, 64), 
    # ('776m-path-dclm-ckpt10000', None, 10000, 64), 

    # ('776m-fope-dclm-v2-ckpt20000', None, 20000, 64), 
    # ('776m-path-dclm-ckpt20000', None, 20000, 64), 

    # ('776m-fope-dclm-v2-ckpt30000', None, 30000, 64), 
    # ('776m-path-dclm-ckpt30000', None, 30000, 64), 
    
    # ('776m-fope-dclm-v2-ckpt40000', None, 40000, 64), 
    # ('776m-path-dclm-ckpt40000', None, 40000, 64), 
    
    # ('776m-fope-dclm-v2-ckpt50000', None, 50000, 64), 
    # ('776m-path-dclm-ckpt50000', None, 50000, 64), 
    
    # ('776m-fope-dclm-v2-ckpt60000', None, 60000, 64), 
    # ('776m-path-dclm-ckpt60000', None, 60000, 64), 
    
    # ('776m-fope-dclm-v2-ckpt80000', None, 80000, 64), 
    # ('776m-path-dclm-ckpt80000', None, 80000, 64), 
    
    # ('776m-fope-dclm-v2-ckpt100000', None, 100000, 64), 
    # ('776m-path-dclm-ckpt100000', None, 100000, 64), 
    
    # ('776m-fope-dclm-v2-ckpt120000', None, 120000, 64), 

    # ('776m-fope-dclm-v2-90000decay', None, 10000, 64), 
    # ('776m-fope-dclm-v2-90000decay-ckpt-ntk3', {'scaling_factor': 3}, 10001, 64), 

]

models = [
    dict(
        type=FoPECausalLM, abbr=abbr, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        max_out_len=max_out_len, batch_size=1, rope_config=rope_config,  # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, rope_config, ckpt, max_out_len in models
]

work_dir = './outputs_xrliu/rope_pp_ruler-v0827/'

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
