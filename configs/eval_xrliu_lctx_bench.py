from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, DLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .dataset_xrliu.core_score import datasets
    from .model_xrliu.lctx_delivery import models

work_dir = './outputs_xrliu/lctx_delivery_bench/'

alillm_h2_cfg = dict(
    bashrc_path='/cpfs01/user/liuxiaoran/.bashrc',
    conda_env_name="/cpfs01/shared/public/changcheng/env/oc_lmdeploy_moe",
    python_env_path="/cpfs01/shared/public/changcheng/env/oc_lmdeploy_moe",
    dlc_config_path="/cpfs01/user/liuxiaoran/dlc.config",
    priority=4,
    workspace_id="ws1os366fx8vi304",  # alillm_h2
    worker_image="pjlab-shanghai-acr-registry-vpc.cn-shanghai.cr.aliyuncs.com/paieflops/chenxun-st:llm-test",
    resource_id='',
    # d-kcs0ludwkhuaav17ve is not available
    data_sources=[],
    driver='535.54.03',
    extra_envs=[
        'HF_HOME=/cpfs01/shared/public/llmeval/hf_cache',
        "HF_DATASETS_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "HF_EVALUATE_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        "NLTK_DATA=/cpfs01/shared/public/llmeval/nltk_data/",
        # "SENTENCE_TRANSFORMERS_HOME=/cpfs01/shared/public/llmeval/torch/sentence_transformers",
        "HF_ENDPOINT=https://hf-mirror.com",

        'PYTHONPATH=./lib/opencompass:$PYTHONPATH',
        'COMPASS_DATA_CACHE=/cpfs01/shared/public/llmeval/compass_data_cache',
    ], 
    dlc_job_cmd='create', 
)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    runner=dict(
        type=DLCRunner,
        max_num_workers=128,
        task=dict(type=OpenICLInferTask),
        aliyun_cfg=alillm_h2_cfg, 
        retry=2),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        max_num_workers=128,
        task=dict(type=OpenICLEvalTask),
        aliyun_cfg=alillm_h2_cfg,
        retry=2),
)


# source /fs-computility/llm/liuxiaoran/.bashrc
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-torch2.4
# python run.py configs/eval_xrliu_lctx_bench.py --dump-eval-details --debug -r 调试用
# python run.py configs/eval_xrliu_lctx_bench.py --dump-eval-details -r 第一次用
#  . 第二次用
