import torch
from mmengine.config import read_base

with read_base():
    # from opencompass.configs.datasets.collections.base_medium_llama import piqa_datasets, siqa_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from opencompass.configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets
    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets
    # from opencompass.configs.datasets.mmlu.mmlu_all_sets import mmlu_all_sets
    # from opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    # from opencompass.configs.datasets.ceval.cev
    # from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets
    from opencompass.configs.datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets
    # from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets
    # from opencompass.configs.datasets.ruler.ruler_niah_gen import niah_datasets
    # from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets

datasets = []
# datasets += hellaswag_datasets
# datasets += winogrande_datasets
# datasets += truthfulqa_datasets
# datasets += mmlu_all_sets
# datasets += ceval_datasets
# datasets += ARC_c_datasets
# datasets += ARC_e_datasets
# datasets += LongBench_2wikimqa_datasets
datasets += ruler_datasets

from opencompass.models import HuggingFaceCausalLM
from opencompass.models.kvcache_work_models.infllm_model import INFLLM_LlamaForCausalLM
from opencompass.models.kvcache_work_models.rag_context_model import RAG_CONTEXT_LlamaForCausalLM
models = [
        dict(
            type=RAG_CONTEXT_LlamaForCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            rag_context_kwargs=dict(
                rag_method='bm25',
                embedding_model_path='/remote-home1/zgliu/models/jina-embeddings-v2-base-zh',
                global_size=64,
                local_size=64,
                text_chunk_size=256,
                text_chunk_overlap=32,
                k=8,
            ),
            tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=819200,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
            abbr='RAG-llama3-8b-64-64-textchunk-256-k-8',
        ),
        dict(
            type=RAG_CONTEXT_LlamaForCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            rag_context_kwargs=dict(
                rag_method='bm25',
                embedding_model_path='/remote-home1/zgliu/models/jina-embeddings-v2-base-zh',
                global_size=64,
                local_size=64,
                text_chunk_size=256,
                text_chunk_overlap=32,
                k=16,
            ),
            tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=819200,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
            abbr='RAG-llama3-8b-64-64-textchunk-256-k-16',
        ),
        dict(
            type=RAG_CONTEXT_LlamaForCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            rag_context_kwargs=dict(
                rag_method='bm25',
                embedding_model_path='/remote-home1/zgliu/models/jina-embeddings-v2-base-zh',
                global_size=64,
                local_size=64,
                text_chunk_size=512,
                text_chunk_overlap=32,
                k=8,
            ),
            tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=819200,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
            abbr='RAG-llama3-8b-64-64-textchunk-512-k-8',
        ),
        dict(
            type=RAG_CONTEXT_LlamaForCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            rag_context_kwargs=dict(
                rag_method='bm25',
                embedding_model_path='/remote-home1/zgliu/models/jina-embeddings-v2-base-zh',
                global_size=32,
                local_size=32,
                text_chunk_size=128,
                text_chunk_overlap=16,
                k=16,
            ),
            tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=819200,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
            abbr='RAG-llama3-8b-32-32-textchunk-128-k-16',
        ),
        dict(
            type=HuggingFaceCausalLM,
            path='/remote-home1/zgliu/models/llama3-8b/',
            model_kwargs=dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16),
            tokenizer_path='/remote-home1/zgliu/models/llama3-8b/',
            tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
            max_seq_len=8192,
            max_out_len=50,
            run_cfg=dict(num_gpus=1, num_procs=1),
            batch_size=1,
            abbr='llama3-8b',
        ),
    ]

from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,  # 最大并行运行进程数
        task=dict(type=OpenICLInferTask),  # 待运行的任务
    )
)