from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPdLLM

with read_base():
    # from opencompass.configs.datasets.ruler.ruler_2k_gen import ruler_datasets as ruler_datasets_2k
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k
    from opencompass.configs.datasets.ruler.ruler_64k_gen import ruler_datasets as ruler_datasets_64k
    from opencompass.configs.datasets.ruler.ruler_96k_gen import ruler_datasets as ruler_datasets_96k

datasets = []
datasets += ruler_datasets_4k
# datasets += ruler_datasets_8k
# datasets += ruler_datasets_16k
# datasets += ruler_datasets_32k
# datasets += ruler_datasets_64k
# datasets += ruler_datasets_96k

num_gpus = {
    'dllm_376m': 1, 'dllm_776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    'dllm_776m-vanilla-dclm': 'rope-0930-776m-llada-4k-vanilla', 
    'dllm_776m-rope_pp_imag1-dclm': 'rope-0930-776m-llada-4k-imag1', 
    'dllm_776m-rope_pp_imag2-dclm': 'rope-0930-776m-llada-4k-imag2', 

}

models = [

    #### 776M

    # ('dllm_776m-vanilla-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 10000, 64), 
    # ('dllm_776m-rope_pp_imag1-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 10000, 64), 
    # ('dllm_776m-rope_pp_imag2-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'steps': 64, 'block_length': 64, }, 10000, 64), 

    # ('dllm_776m-vanilla-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 20000, 64), 
    # ('dllm_776m-rope_pp_imag1-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 20000, 64), 
    # ('dllm_776m-rope_pp_imag2-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'steps': 64, 'block_length': 64, }, 20000, 64), 

    # ('dllm_776m-vanilla-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 30000, 64), 
    # ('dllm_776m-rope_pp_imag1-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 30000, 64), 
    # ('dllm_776m-rope_pp_imag2-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'steps': 64, 'block_length': 64, }, 30000, 64), 
    ('dllm_776m-rope_pp_imag2-dclm-ckpt60000-o32_b32_s32', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'steps': 32, 'block_length': 32, }, 60000, 32), 

]

models = [
    dict(
        type=RoPEPPdLLM, abbr=abbr, 
        rope_config=rope_config, diffusion_config=diffusion_config, seed=2025, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, rope_config, diffusion_config, ckpt, max_out_len in models
]

work_dir = './outputs_xrliu/rope_pp_ruler-dllm/'

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
# python run.py eval_xrliu/eval_xrliu_rope_pp_dllm_ruler.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_dllm_ruler.py --dump-eval-details -r 20240820_190019 第一次用
