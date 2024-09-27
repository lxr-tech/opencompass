from mmengine.config import read_base
from opencompass.models.turbomind_tis import TurboMindTisModel

with read_base():
    # choose a list of datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from opencompass.configs.datasets.race.race_gen_69ee4f import race_datasets
    from opencompass.configs.datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets
    # and output the results in a choosen format
    from opencompass.configs.summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|User|>:', end='\n'),
        dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
    ],
    eos_token_id=103028)

models = [
    dict(
        type=TurboMindTisModel,
        abbr='internlm-chat-20b-turbomind',
        path='internlm',
        tis_addr='0.0.0.0:33337',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
