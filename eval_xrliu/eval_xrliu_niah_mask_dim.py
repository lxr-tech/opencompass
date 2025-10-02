from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import MaskDimCausalLM

with read_base():
    from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_origin_en_datasets
    # from opencompass.configs.datasets.needlebench.needlebench.needlebench import needlebench_parallel_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

is_single_niah = (len([key for key in list(locals()) if key.__contains__('parallel') and key.endswith('datasets')]) == 0)

num_gpus = {
    'llama3_8b': 1, 'llama3_8b_chat': 1, 

    'llama3_2_3b': 1, 'llama3_2_3b_chat': 1, 'llama3_1_8b': 1, 'llama3_1_8b_chat': 1, 

    'qwen3_4b_base': 1, 'qwen3_4b': 1, 'qwen3_8b_base': 1, 'qwen3_8b': 1, 

    'qwen2_5_7b': 1, 'qwen2_5_7b_chat': 1, 'qwen2_5_3b': 1, 'qwen2_5_3b_chat': 1, 
    'qwen2_5_1b': 1, 'qwen2_5_1b_chat': 1, 'qwen2_5_500m': 1, 

    'jamba1_5_mini': 1, 
}

models = [
    # ('llama3_8b', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B', 
    #  None, None, 'llama', ),
    # ('llama3_8b-mask_dim_0_70-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B', 
    #  {'start_dim': 0, 'end_dim': 70, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_8b-mask_dim_70_128-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B', 
    #  {'start_dim': 70, 'end_dim': 128, 'noise_std': 0.8, }, None, 'llama', ),

    # ('llama3_8b-ntk13', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B', 
    #  None, {'ntk_factor': 13}, 'llama', ),
    # ('llama3_8b-ntk13-mask_dim_0_70-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B', 
    #  {'start_dim': 0, 'end_dim': 70, 'noise_std': 0.8}, {'ntk_factor': 13}, 'llama', ),
    # ('llama3_8b-ntk13-mask_dim_70_128-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Meta-Llama-3-8B', 
    #  {'start_dim': 70, 'end_dim': 128, 'noise_std': 0.8}, {'ntk_factor': 13}, 'llama', ),

    # ('llama3_2_3b-mask_dim_0_70-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/', 
    #  {'start_dim': 0, 'end_dim': 70, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_2_3b-mask_dim_70_128-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B/', 
    #  {'start_dim': 70, 'end_dim': 128, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_1_8b-mask_dim_0_70-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/', 
    #  {'start_dim': 0, 'end_dim': 70, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_1_8b-mask_dim_70_128-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B/', 
    #  {'start_dim': 70, 'end_dim': 128, 'noise_std': 0.8, }, None, 'llama', ),

    # ('llama3_2_3b_chat-mask_dim_0_70-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B-Instruct/', 
    #  {'start_dim': 0, 'end_dim': 70, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_2_3b_chat-mask_dim_70_128-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.2-3B-Instruct/', 
    #  {'start_dim': 70, 'end_dim': 128, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_1_8b_chat-mask_dim_0_70-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B-Instruct/', 
    #  {'start_dim': 0, 'end_dim': 70, 'noise_std': 0.8, }, None, 'llama', ),
    # ('llama3_1_8b_chat-mask_dim_70_128-std_0_8', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Llama-3.1-8B-Instruct/', 
    #  {'start_dim': 70, 'end_dim': 128, 'noise_std': 0.8, }, None, 'llama', ),

    # ('qwen3_4b_base-mask_dim_0_92-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B-Base/', 
    #  {'start_dim': 0, 'end_dim': 92, 'noise_std': 1, }, None, 'qwen3', ),
    # ('qwen3_4b_base-mask_dim_92_128-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B-Base/', 
    #  {'start_dim': 92, 'end_dim': 128, 'noise_std': 1, }, None, 'qwen3', ),
    # ('qwen3_8b_base-mask_dim_0_92-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B-Base/', 
    #  {'start_dim': 0, 'end_dim': 92, 'noise_std': 1, }, None, 'qwen3', ),
    # ('qwen3_8b_base-mask_dim_92_128-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B-Base/', 
    #  {'start_dim': 92, 'end_dim': 128, 'noise_std': 1, }, None, 'qwen3', ),

    # ('qwen3_4b-mask_dim_0_92-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B/', 
    #  {'start_dim': 0, 'end_dim': 92, 'noise_std': 1, }, None, 'qwen3', ),
    # ('qwen3_4b-mask_dim_92_128-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-4B/', 
    #  {'start_dim': 92, 'end_dim': 128, 'noise_std': 1, }, None, 'qwen3', ),
    # ('qwen3_8b-mask_dim_0_92-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B/', 
    #  {'start_dim': 0, 'end_dim': 92, 'noise_std': 1, }, None, 'qwen3', ),
    # ('qwen3_8b-mask_dim_92_128-std_1', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/Qwen3-8B/', 
    #  {'start_dim': 92, 'end_dim': 128, 'noise_std': 1, }, None, 'qwen3', ),

    # ('jamba1_5_mini', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/AI21-Jamba-Mini-1.5/', 
    #  None, None, 'jamba', ), 
    ('jamba1_5_mini-mask_dim_0_64-std_4', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/AI21-Jamba-Mini-1.5/', 
     {'start_dim': 0, 'end_dim': 64, 'noise_std': 4, }, None, 'jamba', ), 
    ('jamba1_5_mini-mask_dim_64_128-std_4', '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/downloaded_ckpts/AI21-Jamba-Mini-1.5/', 
     {'start_dim': 64, 'end_dim': 128, 'noise_std': 4, }, None, 'jamba', ), 
]

models = [
    dict(
        type=MaskDimCausalLM, abbr=abbr, path=path, mask_dim_config=mask_dim_config, ntk_config=ntk_config,
        model_type=model_type, model_kwargs={'attn_implementation': 'flash_attention_2'}, 
        max_out_len=50 if is_single_niah else 250, batch_size=1, run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]]),
    ) for abbr, path, mask_dim_config, ntk_config, model_type in models
]

work_dir = './outputs_xrliu/llm_niah/'

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
# conda activate /cpfs01/user/liuxiaoran/miniconda3/envs/llm-cuda12.1
# python run.py eval_xrliu/eval_xrliu_niah_mask_dim.py --dump-eval-details --debug -r  调试用
# python run.py eval_xrliu/eval_xrliu_niah_mask_dim.py --dump-eval-details -r 20240820_190019 第一次用
