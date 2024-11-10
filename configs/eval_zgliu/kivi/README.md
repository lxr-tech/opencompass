https://github.com/jy-yuan/KIVI

conda activate opencompass_eval_kivi

先装kivi依赖再装opencompass

pip install -e .
cd quant && pip install -e .

pip install -e ".[full]"

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

运行评测
```

opencompass kivi/opencompass_eval_kivi.py -w kivi/outputs --debug

```