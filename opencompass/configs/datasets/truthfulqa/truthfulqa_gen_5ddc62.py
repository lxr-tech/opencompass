# from opencompass.openicl.icl_prompt_template import PromptTemplate
# from opencompass.openicl.icl_retriever import ZeroRetriever
# from opencompass.openicl.icl_inferencer import GenInferencer
# from opencompass.datasets import TruthfulQADataset, TruthfulQAEvaluator

# truthfulqa_reader_cfg = dict(
#     input_columns=['question'],
#     output_column='reference',
#     train_split='validation',
#     test_split='validation')

# # TODO: allow empty output-column
# truthfulqa_infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(round=[dict(role='HUMAN', prompt='{question}')])),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer))

# # Metrics such as 'truth' and 'info' needs
# # extra judge models.
# # Please use your own finetuned model  and refers to
# # the source code of `TruthfulQAEvaluator` for more details.

# # If you cannot provide available models for 'truth' and 'info',
# # and want to perform basic metric eval, please set
# # `metrics=('bleurt', 'rouge', 'bleu')`
# truthfulqa_eval_cfg = dict(
#     evaluator=dict(
#         type=TruthfulQAEvaluator, metrics=('bleu'), key='ENV'), )

# truthfulqa_datasets = [
#     dict(
#         abbr='truthful_qa',
#         type=TruthfulQADataset,
#         path='truthful_qa',
#         name='generation',
#         reader_cfg=truthfulqa_reader_cfg,
#         infer_cfg=truthfulqa_infer_cfg,
#         eval_cfg=truthfulqa_eval_cfg)
# ]


from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import AccEvaluator, TruthfulQADatasetForMC


truthfulqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='label',
    train_split='validation',
    test_split='validation')

truthfulqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: f"{{question}} {{A}}", 
            1: f"{{question}} {{B}}", 
            2: f"{{question}} {{C}}", 
            3: f"{{question}} {{D}}", 
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

truthfulqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

truthfulqa_datasets = [
    dict(
        abbr='truthful_qa_mc',
        type=TruthfulQADatasetForMC,
        path='EleutherAI/truthful_qa_mc',
        reader_cfg=truthfulqa_reader_cfg,
        infer_cfg=truthfulqa_infer_cfg,
        eval_cfg=truthfulqa_eval_cfg)
]
