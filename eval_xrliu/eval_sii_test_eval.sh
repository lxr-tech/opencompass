#!/bin/bash

root_path=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089

source ${root_path}/local_conda.sh
echo "start train"
which python
wait
conda activate llamafactory
wait
which python
cd ${root_path}/train/LLaMA-Factory-Cache2State
wait
echo "train start"
sleep 5
# llamafactory-cli train /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/LLaMA-Factory-Cache2State/examples/custome/llama3.2-fla-hybrid-sort2f512_full_pt_ds.yaml
echo "train end"
wait
source ${root_path}/local_conda.sh
echo "start eval"
wait
conda activate fla
wait
which python
cd ${root_path}/train/opencompass
echo "eval start"
sleep 5
# opencompass /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/opencompass/myEval/eval_fla_replace_ckpt-sort2.py
echo "eval end"
