Code for the ACL 2019 paper:

## Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader

Paper link: [https://arxiv.org/abs/1905.07098](https://arxiv.org/abs/1905.07098)

Model Overview:
<p align="center"><img width="90%" src="assets/model.png" /></p>

### Requirements
* ``PyTorch 1.0.1``
* ``tensorboardX``
* ``tqdm``
* ``gluonnlp``

### Prepare data
```
mkdir datasets && cd datasets && wget http://nlp.cs.ucsb.edu/data/webqsp.tar.gz && tar -xzvf webqsp.tar.gz
```

### Full KB setting
**Training**
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id KAReader_full_kb --num_layer 1 --max_num_neighbors 50 --label_smooth 0.1 --data_folder datasets/webqsp/full/ 
```
**Testing**
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id KAReader_full_kb --num_layer 1 --max_num_neighbors 50 --label_smooth 0.1 --data_folder datasets/webqsp/full/ --mode test
```

### Incomplete KB setting (50%)
**Training**
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id KAReader_kb_05 --num_layer 1 --max_num_neighbors 100 --use_doc --label_smooth 0.1 --data_folder datasets/webqsp/kb_05/
```
**Testing**
```
CUDA_VISIBLE_DEVICES=0 python train.py --model_id KAReader_kb_05 --num_layer 1 --max_num_neighbors 100 --use_doc --label_smooth 0.1 --data_folder datasets/webqsp/kb_05/ --mode test --eps 0.12
```

### Bibtex
```
@article{xiong2019improving,
  title={Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader},
  author={Xiong, Wenhan and Yu, Mo and Chang, Shiyu and Guo, Xiaoxiao and Wang, William Yang},
  journal={arXiv preprint arXiv:1905.07098},
  year={2019}
}
```