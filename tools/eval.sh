#!/usr/bin/env bash

# for single card eval
export CUDA_VISIBLE_DEVICES=0
python3.7 tools/eval.py -c ppsci/configs/ldc2d/ldc2d_steady.yaml -o Global.pretrained_model=output/MLP/epoch_200

# for 4-cards eval
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/eval.py -c ppsci/configs/ldc2d/ldc2d_steady.yaml
