from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import MaskCausalLM

# with read_base():
#     from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
#     # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
#     from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

# datasets = []
# datasets += needlebench_origin_en_datasets

# is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)

with read_base():
    # from opencompass.configs.datasets.ruler.ruler_2k_gen import ruler_datasets as ruler_datasets_2k
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k

datasets = []
# datasets += ruler_datasets_2k
datasets += ruler_datasets_4k

num_gpus = {
    '376m': 1, '776m': 1, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-imag2-short': 'rope-0918-376m-4k-imag2-ckpt90000-decay', 
    '376m-imag1-short': 'rope-0918-376m-4k-imag1-ckpt90000-decay', 
    '376m-imag2-long': 'rope-0918-376m-4k-imag2-ckpt90000-decay-lctx', 
    '376m-imag1-long': 'rope-0918-376m-4k-imag1-ckpt90000-decay-lctx', 

    '776m-imag2-short': 'rope-0920-776m-4k-imag2-ckpt90000-decay', 
    '776m-imag1-short': 'rope-0918-776m-4k-imag1-ckpt90000-decay', 
    '776m-imag2-long': 'rope-0920-776m-4k-imag2-ckpt90000-decay-lctx', 
    '776m-imag1-long': 'rope-0918-776m-4k-imag1-ckpt90000-decay-lctx', 

}

models = [

    ## imag1: {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}
    ## imag2: {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}
    
    ('376m-imag2-short-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, None, 10000, 64),

    ('376m-imag2-short-ckpt10000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.2}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.4}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.6}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.8}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.0}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.2}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.5}, 10000, 64),
    ('376m-imag2-short-ckpt10000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 2.0}, 10000, 64),

    ('376m-imag2-short-ckpt10000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.2}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.4}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.6}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.8}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.0}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.2}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.5}, 10000, 64),
    ('376m-imag2-short-ckpt10000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 2.0}, 10000, 64),

    ('376m-imag1-short-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, None, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.2}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.4}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.6}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.8}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.0}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.2}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.5}, 10000, 64),
    ('376m-imag1-short-ckpt10000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 2.0}, 10000, 64),

    ('376m-imag1-short-ckpt10000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.2}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.4}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.6}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.8}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.0}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.2}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.5}, 10000, 64),
    ('376m-imag1-short-ckpt10000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 2.0}, 10000, 64),

    # ('376m-imag2-long-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, None, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.2}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.4}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.6}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.8}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.0}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.2}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.5}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 2.0}, 6000, 64),

    # ('376m-imag2-long-ckpt6000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.2}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.4}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.6}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.8}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.0}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.2}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.5}, 6000, 64),
    # ('376m-imag2-long-ckpt6000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 2.0}, 6000, 64),

    # ('376m-imag1-long-ckpt6000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, None, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.2}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.4}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.6}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.8}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.0}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.2}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.5}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 2.0}, 6000, 64),

    # ('376m-imag1-long-ckpt6000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.2}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.4}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.6}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.8}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.0}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.2}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.5}, 6000, 64),
    # ('376m-imag1-long-ckpt6000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 2.0}, 6000, 64),
    
    # ('776m-imag2-short-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, None, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.2}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.4}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.6}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.8}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.0}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.2}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.5}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 2.0}, 10000, 64),

    # ('776m-imag2-short-ckpt10000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.2}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.4}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.6}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.8}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.0}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.2}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.5}, 10000, 64),
    # ('776m-imag2-short-ckpt10000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 2.0}, 10000, 64),

    # ('776m-imag1-short-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, None, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.2}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.4}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.6}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.8}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.0}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.2}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.5}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 2.0}, 10000, 64),

    # ('776m-imag1-short-ckpt10000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.2}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.4}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.6}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.8}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.0}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.2}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.5}, 10000, 64),
    # ('776m-imag1-short-ckpt10000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 2.0}, 10000, 64),

    # ('776m-imag2-long-ckpt10000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, None, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.2}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.4}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.6}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.8}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.0}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.2}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.5}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 2.0}, 10000, 64),

    # ('776m-imag2-long-ckpt10000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.2}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.4}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.6}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.8}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.0}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.2}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.5}, 10000, 64),
    # ('776m-imag2-long-ckpt10000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 2.0}, 10000, 64),

    # ('776m-imag1-long-ckpt8000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, None, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.2}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.4}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.6}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 0.8}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.0}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.2}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 1.5}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-real_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'real', 'noise_std': 2.0}, 8000, 64),

    # ('776m-imag1-long-ckpt8000-imag_0_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.2}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_0_4', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.4}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_0_6', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.6}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_0_8', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 0.8}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_1_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.0}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_1_2', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.2}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_1_5', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 1.5}, 8000, 64),
    # ('776m-imag1-long-ckpt8000-imag_2_0', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'noise_target': 'imag', 'noise_std': 2.0}, 8000, 64),

]

models = [
    dict(
        type=MaskCausalLM, abbr=abbr, rope_config=rope_config, mask_config = mask_config,
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1,   # max_out_len=64 before 0607
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, rope_config, mask_config, ckpt, max_out_len in models
]


work_dir = './outputs_xrliu/rope_pp_ruler-mask/'

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
        max_num_workers=96, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_niah.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_v0827_niah.py --dump-eval-details -r 20240820_190019 第一次用
