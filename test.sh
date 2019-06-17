#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py --model_id  $2 --num_layer 1 --max_num_neighbors 50 --mode test  --eps 0.08 --data_folder datasets/webqsp/kb_03/