#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py --model_id $2 --num_layer 1 --max_num_neighbors 100 --use_doc --data_folder datasets/webqsp/kb_05/ --eps 0.12