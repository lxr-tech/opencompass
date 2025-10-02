from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPCausalLM_v0827

with read_base():

    from opencompass.configs.datasets.babilong.babilong_0k_gen import babiLong_0k_datasets
    from opencompass.configs.datasets.babilong.babilong_2k_gen import babiLong_2k_datasets
    from opencompass.configs.datasets.babilong.babilong_4k_gen import babiLong_4k_datasets
    from opencompass.configs.datasets.babilong.babilong_8k_gen import babiLong_8k_datasets
    from opencompass.configs.datasets.babilong.babilong_16k_gen import babiLong_16k_datasets
    from opencompass.configs.datasets.babilong.babilong_32k_gen import babiLong_32k_datasets

    from opencompass.configs.datasets.lveval.lveval import LVEval_datasets

datasets = []

# datasets += LVEval_datasets

datasets += babiLong_0k_datasets
datasets += babiLong_2k_datasets
datasets += babiLong_4k_datasets
datasets += babiLong_8k_datasets
# datasets += babiLong_16k_datasets
# datasets += babiLong_32k_datasets

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-vanilla-dclm': 'rope-0904-376m-4k-vanilla', 
    '376m-imagh-dclm': 'rope-0904-376m-4k-imagh', 
    '376m-imago-dclm': 'rope-0904-376m-4k-imago', 
    '376m-fope-dclm': 'rope-0904-376m-4k-fope', 

    '376m-vanilla-dclm-90000decay': 'rope-0904-376m-4k-vanilla-ckpt90000-decay', 
    '376m-imag1-dclm-90000decay': 'rope-0904-376m-4k-imag1-ckpt90000-decay', 
    '376m-imag2-dclm-90000decay': 'rope-0904-376m-4k-imag2-ckpt90000-decay', 
    '376m-imagh-dclm-90000decay': 'rope-0904-376m-4k-imagh-ckpt90000-decay', 
    '376m-imago-dclm-90000decay': 'rope-0904-376m-4k-imago-ckpt90000-decay', 

    '376m-vanilla-dclm-90000decay-lctx': 'rope-0904-376m-4k-vanilla-ckpt90000-decay-lctx', 
    '376m-imagh-dclm-90000decay-lctx': 'rope-0904-376m-4k-imagh-ckpt90000-decay-lctx', 
    '376m-imago-dclm-90000decay-lctx': 'rope-0904-376m-4k-imago-ckpt90000-decay-lctx', 

    '776m-vanilla-fw100b': 'rope-0906-776m-4k-vanilla', 

    '776m-vanilla-dclm-90000decay': 'rope-0904-776m-4k-vanilla-ckpt90000-decay', 
    '776m-imagh-dclm-90000decay': 'rope-0904-776m-4k-imagh-ckpt90000-decay', 
    '776m-imago-dclm-90000decay': 'rope-0904-776m-4k-imago-ckpt90000-decay', 

    '776m-vanilla-dclm-90000decay-lctx': 'rope-0904-776m-4k-vanilla-ckpt90000-decay-lctx', 
    '776m-imagh-dclm-90000decay-lctx': 'rope-0904-776m-4k-imagh-ckpt90000-decay-lctx', 
    '776m-imago-dclm-90000decay-lctx': 'rope-0904-776m-4k-imago-ckpt90000-decay-lctx', 

    '776m-vanilla-dclm-130428decay': 'rope-0904-776m-4k-vanilla-ckpt130428-decay', 
    '776m-imagh-dclm-130428decay': 'rope-0904-776m-4k-imagh-ckpt130428-decay', 
    '776m-imago-dclm-130428decay': 'rope-0904-776m-4k-imago-ckpt130428-decay', 

    '776m-vanilla-dclm': 'rope-0904-776m-4k-vanilla', 
    '776m-imagh-dclm': 'rope-0904-776m-4k-imagh', 
    '776m-imago-dclm': 'rope-0904-776m-4k-imago', 
    '776m-fope-dclm': 'rope-0904-776m-4k-fope', 
    '776m-path-dclm': 'rope-0904-776m-4k-path', 

    '776m-vanilla-fw2-cpt': 'rope-0901-776m-4k-vanilla-ckpt36231-cpt', 
    '776m-imagh-fw2-cpt': 'rope-0901-776m-4k-imagh-ckpt36231-cpt', 
    '776m-imago-fw2-cpt': 'rope-0901-776m-4k-imago-ckpt36231-cpt', 

    '776m-vanilla-fw2': 'rope-0901-776m-4k-vanilla', 
    '776m-imagh-fw2': 'rope-0901-776m-4k-imagh', 
    '776m-imago-fw2': 'rope-0901-776m-4k-imago', 

    '776m-vanilla-fw': 'rope-0831-776m-4k-vanilla', 
    '776m-imagh-fw': 'rope-0831-776m-4k-imagh', 
    '776m-imago-fw': 'rope-0831-776m-4k-imago', 

    # '776m-vanilla': 'rope-0820-776m-4k-vanilla', 
    # '776m-imagh': 'rope-0827-776m-4k-imagh', 
    # '776m-imago': 'rope-0827-776m-4k-imago', 

    # '776m-1dl': 'rope-0827-776m-4k-1dl', 
    # '776m-1d2': 'rope-0820-776m-4k-1d2', 

    # '776m-imag2': 'rope-0820-776m-4k-imag2', 
    # '776m-imag1': 'rope-0820-776m-4k-imag1', 

}


models = [

    ('376m-vanilla-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 30000, 64), 
    ('376m-imag1-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 30000, 64), 
    ('376m-imag2-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 30000, 64), 

    ('376m-vanilla-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 40000, 64), 
    ('376m-imag1-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 40000, 64), 
    ('376m-imag2-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 40000, 64), 

    # ('376m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('376m-imagh-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10001, 64), 
    # ('376m-imago-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10001, 64), 

    # ('776m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('776m-imagh-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10001, 64), 
    # ('776m-imago-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10001, 64), 

    # ('376m-vanilla-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imag1-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imag2-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imagh-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imago-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 10001, 64), 

    # ('776m-vanilla-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('776m-imagh-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 10001, 64), 
    # ('776m-imago-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 10001, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('376m-imagh-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 6000, 64), 
    # ('376m-imago-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 6000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 8000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 8000, 64), 
    # ('776m-imago-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 8000, 64), 

]

models = [
    dict(
        type=RoPEPPCausalLM_v0827, abbr=abbr, rope_config=rope_config, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
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
        max_num_workers=32, 
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
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_babilong.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_babilong.py --dump-eval-details -r 20240820_190019 第一次用
