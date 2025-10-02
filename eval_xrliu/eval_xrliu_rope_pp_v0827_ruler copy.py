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

datasets = []
# datasets += ruler_datasets_2k
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
# datasets += ruler_datasets_16k
# datasets += ruler_datasets_32k

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-vanilla-dclm': 'rope-0904-376m-4k-vanilla', 
    '376m-imag1-dclm': 'rope-0904-376m-4k-imag1', 
    '376m-imagh-dclm': 'rope-0904-376m-4k-imagh', 
    '376m-imago-dclm': 'rope-0904-376m-4k-imago', 
    '376m-fope-dclm': 'rope-0904-376m-4k-fope', 

    '376m-vanilla-dclm-90000decay': 'rope-0904-376m-4k-vanilla-ckpt90000-decay', 
    '376m-imag1-dclm-90000decay': 'rope-0904-376m-4k-imag1-ckpt90000-decay', 
    '376m-imag2-dclm-90000decay': 'rope-0904-376m-4k-imag2-ckpt90000-decay', 
    '376m-imagh-dclm-90000decay': 'rope-0904-376m-4k-imagh-ckpt90000-decay', 
    '376m-imago-dclm-90000decay': 'rope-0904-376m-4k-imago-ckpt90000-decay', 

    '376m-vanilla-dclm-90000decay-lctx': 'rope-0904-376m-4k-vanilla-ckpt90000-decay-lctx', 
    '376m-imag1-dclm-90000decay-lctx': 'rope-0904-376m-4k-imag1-ckpt90000-decay-lctx', 
    '376m-imag2-dclm-90000decay-lctx': 'rope-0904-376m-4k-imag2-ckpt90000-decay-lctx', 
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

    # ('376m-vanilla-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('376m-imagh-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 40000, 64), 
    # ('376m-imago-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 40000, 64), 

    # ('376m-vanilla-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('376m-imagh-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 60000, 64), 
    # ('376m-imago-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 60000, 64), 

    # ('376m-vanilla-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('376m-imagh-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 80000, 64), 
    # ('376m-imago-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 80000, 64), 

    # ('376m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('376m-imag1-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('376m-imag2-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 10001, 64), 
    # ('376m-imagh-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10001, 64), 
    # ('376m-imago-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10001, 64), 

    # ('376m-vanilla-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imag1-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imag2-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imagh-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 10001, 64), 
    # ('376m-imago-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 10001, 64), 

    ('376m-vanilla-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 2000, 64), 
    ('376m-imag1-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 2000, 64), 
    ('376m-imag2-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 2000, 64), 
    # ('376m-imagh-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 2000, 64), 
    # ('376m-imago-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 2000, 64), 

    ('376m-vanilla-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 4000, 64), 
    ('376m-imag1-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 4000, 64), 
    ('376m-imag2-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 4000, 64), 
    # ('376m-imagh-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 4000, 64), 
    # ('376m-imago-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 4000, 64), 

    ('376m-vanilla-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 
    ('376m-imag1-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 6000, 64), 
    ('376m-imag2-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 6000, 64), 
    # ('376m-imagh-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 6000, 64), 
    # ('376m-imago-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 6000, 64), 

    ('376m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 8000, 64), 
    ('376m-imag1-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 8000, 64), 
    ('376m-imag2-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 8000, 64), 
    # ('376m-imagh-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 8000, 64), 
    # ('376m-imago-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 8000, 64), 

    # ('376m-vanilla-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-imag1-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('376m-imagh-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10000, 64), 
    # ('376m-imago-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10000, 64), 

    # ('776m-vanilla-fw100b-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-vanilla-fw100b-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 20000, 64), 
    # ('776m-vanilla-fw100b-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('776m-vanilla-fw100b-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('776m-vanilla-fw100b-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 50000, 64), 
    # ('776m-vanilla-fw100b-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('776m-vanilla-fw100b-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('776m-vanilla-fw100b-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-imagh-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10000, 64), 
    # ('776m-imago-dclm-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10000, 64), 
    # ('776m-fope-dclm-ckpt10000', None, 10000, 64), 
    # ('776m-path-dclm-ckpt10000', None, 10000, 64), 

    # ('776m-vanilla-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 20000, 64), 
    # ('776m-imagh-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 20000, 64), 
    # ('776m-imago-dclm-ckpt20000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 20000, 64), 
    # ('776m-fope-dclm-ckpt20000', None, 20000, 64), 
    # ('776m-path-dclm-ckpt20000', None, 20000, 64), 

    # ('776m-vanilla-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 30000, 64), 
    # ('776m-imagh-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 30000, 64), 
    # ('776m-imago-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 30000, 64), 
    # ('776m-fope-dclm-ckpt30000', None, 30000, 64), 
    # ('776m-path-dclm-ckpt30000', None, 30000, 64), 

    # ('776m-vanilla-dclm-ckpt36231', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 
    # ('776m-imagh-dclm-ckpt36231', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 36231, 64), 
    # ('776m-imago-dclm-ckpt36231', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 36231, 64), 

    # ('776m-vanilla-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 40000, 64), 
    # ('776m-imagh-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 40000, 64), 
    # ('776m-imago-dclm-ckpt40000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 40000, 64), 
    # ('776m-fope-dclm-ckpt40000', None, 40000, 64), 
    # ('776m-path-dclm-ckpt40000', None, 40000, 64), 

    # ('776m-vanilla-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 50000, 64), 
    # ('776m-imagh-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 50000, 64), 
    # ('776m-imago-dclm-ckpt50000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 50000, 64), 

    # ('776m-vanilla-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 60000, 64), 
    # ('776m-imagh-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 60000, 64), 
    # ('776m-imago-dclm-ckpt60000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 60000, 64), 

    # ('776m-vanilla-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 80000, 64), 
    # ('776m-imagh-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 80000, 64), 
    # ('776m-imago-dclm-ckpt80000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 80000, 64), 

    # ('776m-vanilla-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 100000, 64), 
    # ('776m-imagh-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 100000, 64), 
    # ('776m-imago-dclm-ckpt100000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 100000, 64), 
    # ('776m-imagh-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 100000, 64), 
    # ('776m-imago-dclm-ckpt100000-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt100000-ntk8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 8}, 100000, 64), 
    # ('776m-imagh-dclm-ckpt100000-ntk8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 8}, 100000, 64), 
    # ('776m-imago-dclm-ckpt100000-ntk8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 8}, 100000, 64), 

    # ('776m-vanilla-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 120000, 64), 
    # ('776m-imagh-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 120000, 64), 
    # ('776m-imago-dclm-ckpt120000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 120000, 64), 

    # ('776m-vanilla-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10001, 64), 
    # ('776m-imagh-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10001, 64), 
    # ('776m-imago-dclm-90000decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10001, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 2000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 2000, 64), 
    # ('776m-imago-dclm-90000decay-lctx-ckpt2000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 2000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 4000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 4000, 64), 
    # ('776m-imago-dclm-90000decay-lctx-ckpt4000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 4000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 6000, 64), 
    # ('776m-imago-dclm-90000decay-lctx-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 6000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 6000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 8000, 64), 
    # ('776m-imago-dclm-90000decay-lctx-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 8000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10000, 64), 
    # ('776m-imago-dclm-90000decay-lctx-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10000, 64), 

    # ('776m-vanilla-dclm-90000decay-lctx', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 10000, 64), 
    # ('776m-imagh-dclm-90000decay-lctx', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 10000, 64), 
    # ('776m-imago-dclm-90000decay-lctx', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 10000, 64), 

    # ('776m-vanilla-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 10001, 64), 
    # ('776m-imagh-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 10001, 64), 
    # ('776m-imago-dclm-90000decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 10001, 64), 

    # ('776m-vanilla-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 14493, 64), 
    # ('776m-imagh-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 14493, 64), 
    # ('776m-imago-dclm-130428decay', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 14493, 64), 

    # ('776m-vanilla-dclm-130428decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1', 'scaling_factor': 3}, 14493, 64), 
    # ('776m-imagh-dclm-130428decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh', 'scaling_factor': 3}, 14493, 64), 
    # ('776m-imago-dclm-130428decay-ckpt-ntk3', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago', 'scaling_factor': 3}, 14493, 64), 

]
"""
    # ('776m-vanilla-fw2-cpt-ckpt94197', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 94197, 64), 
    # ('776m-imagh-fw2-cpt-ckpt94197', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 94197, 64), 
    # ('776m-imago-fw2-cpt-ckpt94197', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 94197, 64), 

    # ('776m-vanilla-fw2-cpt-ckpt78000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 78000, 64), 
    # ('776m-imagh-fw2-cpt-ckpt78000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 78000, 64), 
    # ('776m-imago-fw2-cpt-ckpt78000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 78000, 64), 

    # ('776m-vanilla-fw2-cpt-ckpt52000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 52000, 64), 
    # ('776m-imagh-fw2-cpt-ckpt52000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 52000, 64), 
    # ('776m-imago-fw2-cpt-ckpt52000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 52000, 64), 

    # ('776m-vanilla-fw2-cpt-ckpt39000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 39000, 64), 
    # ('776m-imagh-fw2-cpt-ckpt39000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 39000, 64), 
    # ('776m-imago-fw2-cpt-ckpt39000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 39000, 64), 

    # ('776m-vanilla-fw2-cpt-ckpt26000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 26000, 64), 
    # ('776m-imagh-fw2-cpt-ckpt26000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 26000, 64), 
    # ('776m-imago-fw2-cpt-ckpt26000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 26000, 64), 

    # ('776m-vanilla-fw2-cpt-ckpt100', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 100, 64), 
    # ('776m-imagh-fw2-cpt-ckpt100', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 100, 64), 
    # ('776m-imago-fw2-cpt-ckpt100', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 100, 64), 

    # ('776m-vanilla-fw2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 
    # ('776m-imagh-fw2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 36231, 64), 
    # ('776m-imago-fw2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 36231, 64), 

    # ('776m-vanilla-fw2-ckpt18000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 18000, 64), 
    # ('776m-imagh-fw2-ckpt18000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 18000, 64), 
    # ('776m-imago-fw2-ckpt18000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 18000, 64), 

    # ('776m-vanilla-fw2-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 9058, 64), 
    # ('776m-imagh-fw2-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 9058, 64), 
    # ('776m-imago-fw2-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 9058, 64), 

    # ('776m-vanilla-fw', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 
    # ('776m-imagh-fw', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 36231, 64), 
    # ('776m-imago-fw', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 36231, 64), 

    # ('776m-vanilla-fw-ckpt18000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 18000, 64), 
    # ('776m-imagh-fw-ckpt18000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 18000, 64), 
    # ('776m-imago-fw-ckpt18000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 18000, 64), 

    # ('776m-vanilla-fw-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 9058, 64), 
    # ('776m-imagh-fw-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 9058, 64), 
    # ('776m-imago-fw-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 9058, 64), 

    # # ('776m-vanilla', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 
    # # ('776m-imagh', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 36231, 64), 
    # # ('776m-imago', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 36231, 64), 
    
    # ('776m-vanilla-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 9058, 64), 
    # ('776m-imagh-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imagh'}, 9058, 64), 
    # ('776m-imago-ckpt9058', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imago'}, 9058, 64), 

    # # ('776m-1d2-ckpt9058', {'1d': True, '1d_mode': '1d2', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 9058, 64), 
    # # ('776m-1dl-ckpt9058', {'1d': True, '1d_mode': '1dl', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 9058, 64), 

    # ('776m-1d2', {'1d': True, '1d_mode': '1d2', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 
    # ('776m-1dl', {'1d': True, '1d_mode': '1dl', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, 36231, 64), 

    # # ('776m-imag2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, 36231, 64), 
    # # ('776m-imag1', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, 36231, 64), 
"""

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
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_ruler.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_ruler.py --dump-eval-details -r 20240820_190019 第一次用
