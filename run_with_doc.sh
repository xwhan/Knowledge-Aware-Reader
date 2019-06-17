#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py --model_id $2 --num_layer 1 --max_num_neighbors 100 --use_doc --label_smooth 0.1 --data_folder datasets/webqsp/full/