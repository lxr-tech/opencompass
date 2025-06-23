#!/bin/bash

source /inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/public/resurrection.sh
wait
conda activate llm-c2s
wait
cd /inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_xrliu/opencompass
wait
python run.py eval_xrliu/eval_xrliu_niah_mask_dim.py --dump-eval-details -r
