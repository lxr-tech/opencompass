from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import FoPECausalLM

with read_base():
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc import gsm8k_datasets  # noqa: F401, F403
    # # from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_250710 import humaneval_datasets as humaneval_xrliu_datasets
    # # from opencompass.configs.datasets.mbpp.deprecated_sanitized_mbpp_passk_gen_1e1056 import sanitized_mbpp_datasets

    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen_5ddc62 import truthfulqa_datasets  # noqa: F401, F403

    from opencompass.configs.datasets.lambada.lambada_gen_217e11 import lambada_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets  # zero shot

    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets

    from opencompass.configs.datasets.gpqa.gpqa_ppl_6bf57a import gpqa_datasets
    from opencompass.configs.datasets.siqa.siqa_ppl_ced5f6 import siqa_datasets
    from opencompass.configs.datasets.obqa.obqa_ppl_c7c154 import obqa_datasets

    from opencompass.configs.datasets.nq.nq_gen_3dcea1 import nq_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    # from opencompass.configs.datasets.commonsenseqa.commonsenseqa_ppl_5545e2 import commonsenseqa_datasets  # buggy

    from opencompass.configs.datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_ppl import AX_b_datasets
    from opencompass.configs.datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_ppl import AX_g_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import BoolQ_datasets
    from opencompass.configs.datasets.SuperGLUE_CB.SuperGLUE_CB_ppl import CB_datasets
    from opencompass.configs.datasets.SuperGLUE_COPA.SuperGLUE_COPA_ppl import COPA_datasets
    from opencompass.configs.datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_ppl import MultiRC_datasets
    from opencompass.configs.datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen import ReCoRD_datasets
    from opencompass.configs.datasets.SuperGLUE_RTE.SuperGLUE_RTE_ppl import RTE_datasets
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_ppl import WiC_datasets
    from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl import WSC_datasets
 
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

batch_size = {
    '376m': 16, '776m': 16, 
}

path_root = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/rope_pp/checkpoints'

path_dict = {

    '376m-fope-dclm-v2': 'rope-0918-376m-4k-fope', 
    '376m-fope-dclm-v2-90000decay': 'rope-0918-376m-4k-fope-ckpt90000-decay', 

    '776m-fope-dclm-v2': 'rope-0918-776m-4k-fope', 
    '776m-fope-dclm-v2-90000decay': 'rope-0918-776m-4k-fope-ckpt90000-decay', 

}

models = [

    # ('376m-fope-dclm-v2-ckpt10000', None, 10000, 64), 
    # ('376m-fope-dclm-v2-ckpt20000', None, 20000, 64), 
    # ('376m-fope-dclm-v2-ckpt30000', None, 30000, 64), 
    # ('376m-fope-dclm-v2-ckpt40000', None, 40000, 64), 
    # ('376m-fope-dclm-v2-ckpt50000', None, 50000, 64), 
    # ('376m-fope-dclm-v2-ckpt60000', None, 60000, 64), 
    # ('376m-fope-dclm-v2-ckpt80000', None, 80000, 64), 
    # ('376m-fope-dclm-v2-ckpt100000', None, 100000, 64), 
    # ('376m-fope-dclm-v2-ckpt120000', None, 120000, 64), 

    # ('376m-fope-dclm-v2-40000decay-ckpt5000', None, 5000, 64), 
    # ('376m-fope-dclm-v2-40000decay', None, 10000, 64), 
    # ('376m-fope-dclm-v2-90000decay', None, 10000, 64), 
    # ('376m-fope-dclm-v2-90000decay-ckpt-ntk3', {'scaling_factor': 3}, 10001, 64), 

    # ('776m-fope-dclm-v2-ckpt10000', None, 10000, 64), 
    # ('776m-fope-dclm-v2-ckpt20000', None, 20000, 64), 
    # ('776m-fope-dclm-v2-ckpt30000', None, 30000, 64), 
    # ('776m-fope-dclm-v2-ckpt40000', None, 40000, 64), 
    # ('776m-fope-dclm-v2-ckpt50000', None, 50000, 64), 
    # ('776m-fope-dclm-v2-ckpt60000', None, 60000, 64), 
    # ('776m-fope-dclm-v2-ckpt80000', None, 80000, 64), 
    ('776m-fope-dclm-v2-ckpt100000', None, 100000, 64), 
    # ('776m-fope-dclm-v2-ckpt120000', None, 120000, 64), 

    # ('776m-fope-dclm-v2-90000decay', None, 10000, 64), 
    # ('776m-fope-dclm-v2-90000decay-ckpt-ntk3', {'scaling_factor': 3}, 10001, 64), 

]

models = [
    dict(
        type=FoPECausalLM, abbr=abbr, 
        path=f"{path_root}/{path_dict[abbr.split('-ckpt')[0]]}/checkpoint-{ckpt}", 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, max_seq_len=2048,  # added in 09061436
        batch_size=batch_size[abbr.split('-')[0]], run_cfg=dict(num_gpus=1),
    ) for abbr, rope_config, ckpt, max_out_len in models
]

work_dir = './outputs_xrliu/rope_pp_short-v0827/'

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
        max_num_workers=32, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

 
# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate llm-fepe
# python run.py eval_xrliu/eval_xrliu_rope_pp_short.py --dump-eval-details -r --debug  调试用
# python run.py eval_xrliu/eval_xrliu_rope_pp_short.py --dump-eval-details -r 20240820_190019 第一次用
