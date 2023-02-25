#!/usr/bin/env bash

# for single card train
export CUDA_VISIBLE_DEVICES=4
python3.7 tools/train.py -c ppsci/configs/ldc2d/ldc2d_steady.yaml

# for 4-cards train
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ppsci/configs/ldc2d/ldc2d_steady.yaml
