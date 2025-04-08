from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.datasets import LongBenchv2Dataset, AccEvaluator

LongBenchv2_reader_cfg = dict(
    input_columns=['context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'difficulty', 'length'],
    output_column='answer',
)

LongBenchv2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'A':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Please read the following text and answer the questions below.\n <text> \n {context} \n </text> \n \n Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{choice_A}')
                ], ),
            'B':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Please read the following text and answer the questions below.\n <text> \n {context} \n </text> \n \n Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{choice_B}')
                ], ),
            'C':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Please read the following text and answer the questions below.\n <text> \n {context} \n </text> \n \n Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{choice_C}')
                ], ),
            'D':
            dict(
                round=[
                    dict(role='HUMAN', prompt='Please read the following text and answer the questions below.\n <text> \n {context} \n </text> \n \n Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{choice_D}')
                ], ),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))
# dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role='HUMAN',
#                     prompt='Please read the following text and answer the questions below.\n <text> \n {context} \n </text> \n \n What is the correct answer to this question: {question} \n \n Choices: \n (A) {choice_A} \n (B) {choice_B} \n (C) {choice_C} \n (D) {choice_D} \n Let’s think step by step. Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)',
#                 ),
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer),
# )

LongBenchv2_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

LongBenchv2_datasets = [
    dict(
        type=LongBenchv2Dataset,
        abbr='LongBenchv2_ppl_mc',
        path='opencompass/longbenchv2',
        reader_cfg=LongBenchv2_reader_cfg,
        infer_cfg=LongBenchv2_infer_cfg,
        eval_cfg=LongBenchv2_eval_cfg,
    )
]
