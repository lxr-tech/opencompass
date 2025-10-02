from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPCausalLM_v0827

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
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
datasets += ruler_datasets_32k
datasets += ruler_datasets_64k
# datasets += ruler_datasets_96k

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-vanilla-dclm': 'rope-0904-376m-4k-vanilla', 
    '376m-rope_pp_imag1-dclm': 'rope-0918-376m-4k-imag1', 
    '376m-rope_pp_imag2-dclm': 'rope-0918-376m-4k-imag2', 
    '376m-fope-dclm': 'rope-0904-376m-4k-fope', 

    '376m-vanilla-dclm-90000decay': 'rope-0904-376m-4k-vanilla-ckpt90000-decay', 
    '376m-rope_pp_imag1-dclm-90000decay': 'rope-0918-376m-4k-imag1-ckpt90000-decay', 
    '376m-rope_pp_imag2-dclm-90000decay': 'rope-0918-376m-4k-imag2-ckpt90000-decay', 

    '376m-vanilla-dclm-90000decay-lctx': 'rope-0904-376m-4k-vanilla-ckpt90000-decay-lctx', 
    '376m-rope_pp_imag1-dclm-90000decay-lctx': 'rope-0918-376m-4k-imag1-ckpt90000-decay-lctx', 
    '376m-rope_pp_imag2-dclm-90000decay-lctx': 'rope-0918-376m-4k-imag2-ckpt90000-decay-lctx', 

    '776m-vanilla-dclm': 'rope-0904-776m-4k-vanilla', 
    '776m-rope_pp_imag1-dclm': 'rope-0918-776m-4k-imag1', 
    '776m-rope_pp_imag2-dclm': 'rope-0918-776m-4k-imag2', 
    '776m-vanilla-dclm-v2': 'rope-0920-776m-4k-vanilla', 
    '776m-rope_pp_imag1-dclm-v2': 'rope-0920-776m-4k-imag1', 
    '776m-rope_pp_imag2-dclm-v2': 'rope-0920-776m-4k-imag2', 
    '776m-path-dclm': 'rope-0904-776m-4k-path', 
    '776m-fope-dclm': 'rope-0904-776m-4k-fope', 

    '776m-vanilla-dclm-90000decay': 'rope-0904-776m-4k-vanilla-ckpt90000-decay', 
    '776m-rope_pp_imag1-dclm-90000decay': 'rope-0918-776m-4k-imag1-ckpt90000-decay', 
    '776m-rope_pp_imag2-dclm-90000decay': 'rope-0918-776m-4k-imag2-ckpt90000-decay', 
    '776m-rope_pp_imag1-dclm-90000decay-v2': 'rope-0918-776m-4k-imag1-ckpt90000-decay', 
    '776m-rope_pp_imag2-dclm-90000decay-v2': 'rope-0918-776m-4k-imag2-ckpt90000-decay', 
    '776m-rope_pp_imag1-dclm-v2-90000decay': 'rope-0920-776m-4k-imag1-ckpt90000-decay', 
    '776m-rope_pp_imag2-dclm-v2-90000decay': 'rope-0920-776m-4k-imag2-ckpt90000-decay', 

    '776m-vanilla-dclm-90000decay-lctx': 'rope-0904-776m-4k-vanilla-ckpt90000-decay-lctx', 
    '776m-rope_pp_imag1-dclm-90000decay-lctx': 'rope-0918-776m-4k-imag1-ckpt90000-decay-lctx', 
    '776m-rope_pp_imag2-dclm-90000decay-lctx': 'rope-0918-776m-4k-imag2-ckpt90000-decay-lctx', 
    '776m-rope_pp_imag1-dclm-v2-90000decay-lctx': 'rope-0920-776m-4k-imag1-ckpt90000-decay-lctx', 
    '776m-rope_pp_imag2-dclm-v2-90000decay-lctx': 'rope-0920-776m-4k-imag2-ckpt90000-decay-lctx', 

}

models = [

    # ('376m-vanilla-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 

    # ('376m-vanilla-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 20000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 20000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 20000, 64), 

    # ('376m-vanilla-dclm-ckpt26946', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 26946, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt26946', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 26946, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt26946', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 26946, 64), 

    # ('376m-vanilla-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 30000, 64), 

    # ('376m-vanilla-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 40000, 64), 

    # ('376m-vanilla-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 50000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 50000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 50000, 64), 

    # ('376m-vanilla-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 60000, 64), 

    # ('376m-vanilla-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 80000, 64), 

    # ('376m-vanilla-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 100000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 100000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 100000, 64), 

    # ('376m-vanilla-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 100000, 64), 
    # ('376m-rope_pp_imag1-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1', 'scaling_factor': 3}, 100000, 64), 
    # ('376m-rope_pp_imag2-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2', 'scaling_factor': 3}, 100000, 64), 

    # ('376m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 

    # ('376m-vanilla-dclm-90000decay-ckpt-ntk2_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 2.5}, 10001, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-ckpt-ntk2_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1', 'scaling_factor': 2.5}, 10000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-ckpt-ntk2_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2', 'scaling_factor': 2.5}, 10000, 64), 

    # ('376m-vanilla-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2', 'scaling_factor': 3}, 10000, 64), 

    # ('376m-vanilla-dclm-90000decay-ckpt-ntk6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 6}, 10001, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-ckpt-ntk6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1', 'scaling_factor': 6}, 10000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-ckpt-ntk6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2', 'scaling_factor': 6}, 10000, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 2000, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 2000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 2000, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 4000, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 4000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 4000, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 6000, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 8000, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 8000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 8000, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-rope_pp_imag1-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-rope_pp_imag2-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 

    #### 776M

    # ('776m-vanilla-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 
    # # ('776m-fope-dclm-ckpt10000', None, 10000, 64), 
    # # ('776m-path-dclm-ckpt10000', None, 10000, 64), 

    # ('776m-vanilla-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 20000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 20000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 20000, 64), 
    # # ('776m-fope-dclm-ckpt20000', None, 20000, 64), 
    # # ('776m-path-dclm-ckpt20000', None, 20000, 64), 

    # ('776m-vanilla-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 30000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 30000, 64), 
    # # ('776m-fope-dclm-ckpt30000', None, 30000, 64), 
    # # ('776m-path-dclm-ckpt30000', None, 30000, 64), 

    # ('776m-vanilla-dclm-ckpt36231', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt36231', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 36231, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt36231', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 36231, 64), 

    # ('776m-vanilla-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 40000, 64), 
    # # ('776m-fope-dclm-ckpt40000', None, 40000, 64), 
    # # ('776m-path-dclm-ckpt40000', None, 40000, 64), 

    # ('776m-vanilla-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 50000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 50000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 50000, 64), 

    # ('776m-vanilla-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 60000, 64), 
    # ('776m-vanilla-dclm-v2-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 60000, 64), 

    # ('776m-vanilla-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 80000, 64), 
    # ('776m-vanilla-dclm-v2-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 80000, 64), 

    # ('776m-vanilla-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 100000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 100000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 100000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 120000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 120000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 120000, 64), 

    # ('776m-vanilla-dclm-ckpt140000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 140000, 64), 
    # ('776m-rope_pp_imag1-dclm-ckpt140000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 140000, 64), 
    # ('776m-rope_pp_imag2-dclm-ckpt140000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 140000, 64), 

    # ('776m-vanilla-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt100000-ntk8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 8}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 120000, 64), 

    # ('776m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 
    # ('776m-rope_pp_imag1-dclm-90000decay-v2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag2-dclm-90000decay-v2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 

    # # ('776m-vanilla-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 2000, 64), 
    # ('776m-rope_pp_imag1-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 2000, 64), 
    # ('776m-rope_pp_imag2-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 2000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 4000, 64), 
    # ('776m-rope_pp_imag1-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 4000, 64), 
    # ('776m-rope_pp_imag2-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 4000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('776m-rope_pp_imag1-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('776m-rope_pp_imag2-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 6000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 6000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 8000, 64), 
    # ('776m-rope_pp_imag1-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 8000, 64), 
    # ('776m-rope_pp_imag2-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 8000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 8000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 8000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag1-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag2-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 
    # ('776m-rope_pp_imag1-dclm-v2-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-rope_pp_imag2-dclm-v2-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 

    # ('776m-vanilla-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 14493, 64), 

    # ('776m-vanilla-dclm-130428decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 14493, 64), 

]

models = [
    dict(
        type=RoPEPPCausalLM_v0827, abbr=abbr, rope_config=rope_config, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
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
