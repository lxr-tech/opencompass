from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPdLLM

with read_base():
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.math.math_500_gen import math_datasets
    # # from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_250710 import humaneval_datasets as humaneval_xrliu_datasets
    # # from opencompass.configs.datasets.mbpp.deprecated_sanitized_mbpp_passk_gen_1e1056 import sanitized_mbpp_datasets

    # from opencompass.configs.datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets  # noqa: F401, F403

    # from opencompass.configs.datasets.lambada.lambada_gen import lambada_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.hellaswag.hellaswag_gen import hellaswag_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.winogrande.winogrande_gen import winogrande_datasets  # zero shot

    # from opencompass.configs.datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    # from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

    # # from opencompass.configs.datasets.gpqa.gpqa_gen_5shot import gpqa_datasets
    # from opencompass.configs.datasets.siqa.siqa_gen import siqa_datasets
    # from opencompass.configs.datasets.obqa.obqa_gen import obqa_datasets

    # from opencompass.configs.datasets.nq.nq_gen_3dcea1 import nq_datasets
    # from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets

    # from opencompass.configs.datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_gen import AX_b_datasets
    # from opencompass.configs.datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_gen import AX_g_datasets
    # from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen import BoolQ_datasets
    # from opencompass.configs.datasets.SuperGLUE_CB.SuperGLUE_CB_gen import CB_datasets
    # from opencompass.configs.datasets.SuperGLUE_COPA.SuperGLUE_COPA_gen import COPA_datasets
    # from opencompass.configs.datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_gen import MultiRC_datasets
    # from opencompass.configs.datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen import ReCoRD_datasets
    # from opencompass.configs.datasets.SuperGLUE_RTE.SuperGLUE_RTE_gen import RTE_datasets
    # from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen import WiC_datasets
    # from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen import WSC_datasets
 
    # from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import mmlu_datasets


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

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

    ('dllm_776m-vanilla-dclm-ckpt2038-o32_b32_s32', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': False, 'imag_mode': 'imag1'}, {'steps': 32, 'block_length': 32, }, 2038, 32), 
    # ('dllm_776m-rope_pp_imag1-dclm-ckpt30000', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag1'}, {'steps': 64, 'block_length': 64, }, 30000, 64), 
    # ('dllm_776m-rope_pp_imag2-dclm-ckpt60000-o32_b32_s32', {'1d': False, '1d_mode': '1d1', 'imp': None, 'imp_mode': 'partial', 'imag': True, 'imag_mode': 'imag2'}, {'steps': 32, 'block_length': 32, }, 60000, 32), 

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

work_dir = './outputs_xrliu/rope_pp_short-dllm/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=4000, gen_task_coef=15),
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

 
# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate llm-fepe
# python run.py eval_xrliu/eval_xrliu_rope_pp_dllm_short.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_dllm_short.py --dump-eval-details -r 20240820_190019 第一次用
