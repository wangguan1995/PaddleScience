export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch --gpus="6,7" examples/laplace/laplace2d.py