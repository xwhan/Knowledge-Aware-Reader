import json
import nltk
import numpy as np
import os
import torch
import yaml

from collections import Counter

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from tqdm import tqdm

import argparse

def get_config(config_path=None):
    if not config_path:
        parser = argparse.ArgumentParser()

        # datasets
        parser.add_argument('--name', default='webqsp', type=str)
        parser.add_argument('--data_folder', default='datasets/webqsp/kb_03/', type=str)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--train_documents', default='documents.json', type=str)
        parser.add_argument('--dev_data', default='dev.json', type=str)
        parser.add_argument('--dev_documents', default='documents.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--test_documents', default='documents.json', type=str)
        parser.add_argument('--max_query_word', default=10, type=int)
        parser.add_argument('--max_document_word', default=50, type=int)
        parser.add_argument('--max_char', default=25, type=int)
        parser.add_argument('--max_num_neighbors', default=100, type=int)
        parser.add_argument('--max_rel_words', default=8, type=int)

        # embeddings
        parser.add_argument('--word2id', default='glove_vocab.txt', type=str)
        parser.add_argument('--relation2id', default='relations.txt', type=str)
        parser.add_argument('--entity2id', default='entities.txt', type=str)
        parser.add_argument('--char2id', default='chars.txt', type=str)
        parser.add_argument('--word_emb_file', default='glove_word_emb.npy', type=str)
        parser.add_argument('--entity_emb_file', default='entity_emb_100d.npy', type=str)
        parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)

        # dimensions, layers, dropout
        parser.add_argument('--num_layer', default=1, type=int)
        parser.add_argument('--entity_dim', default=100, type=int)
        parser.add_argument('--word_dim', default=300, type=int)
        parser.add_argument('--hidden_drop', default=0.2, type=float)
        parser.add_argument('--word_drop', default=0.2, type=float)

        # optimization
        parser.add_argument('--num_epoch', default=100, type=int)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--gradient_clip', default=1.0, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--seed', default=19940715, type=int)
        parser.add_argument('--lr_schedule', action='store_true')
        parser.add_argument('--label_smooth', default=0.1, type=float)
        parser.add_argument('--fact_drop', default=0, type=float)

        # model options
        parser.add_argument('--use_doc', action='store_true')
        parser.add_argument('--use_kb', action='store_true')
        parser.add_argument('--use_inverse_relation', action='store_true')
        parser.add_argument('--model_id', default='debug', type=str)
        parser.add_argument('--load_model_file', default=None, type=str)
        parser.add_argument('--mode', default='train', type=str)
        parser.add_argument('--eps', default=0.05, type=float) # threshold for f1

        args = parser.parse_args()

        if args.name == 'webqsp':
            args.type_rels = ['<fb:food.dish.type_of_dish1>', '<fb:film.performance.special_performance_type>', '<fb:geography.mountain.mountain_type>', '<fb:base.aareas.schema.administrative_area.administrative_area_type>', '<fb:base.qualia.disability.type_of_disability>', '<fb:common.topic.notable_types>', '<fb:base.events.event_feed.type_of_event>', '<fb:base.disaster2.injury.type_of_event>', '<fb:religion.religion.types_of_places_of_worship>', '<fb:tv.tv_regular_personal_appearance.appearance_type>']
        else:
            args.type_rels = []

        config = vars(args)
        config['to_save_model'] = True # always save model
        config['save_model_file'] = 'model/' + config['name'] + '/best_{}.pt'.format(config['model_id'])
        config['pred_file'] = 'results/' + config['name'] + '/best_{}.pred'.format(config['model_id'])
    else:
        with open(config_path, "r") as setting:
            config = yaml.load(setting)

    print('-'* 10 + 'Experiment Config' + '-' * 10)
    for k, v in config.items():
        print(k + ': ', v)
    print('-'* 10 + 'Experiment Config' + '-' * 10 + '\n')

    return config

def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def save_model(the_model, path):
    if os.path.exists(path):
        path = path + '_copy'
    print("saving model to ...", path)
    torch.save(the_model, path)


def load_model(path):
    if not os.path.exists(path):
        assert False, 'cannot find model: ' + path
    print("loading model from ...", path)
    return torch.load(path)

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def load_documents(document_file):
    print('loading document from', document_file)
    documents = dict()
    with open(document_file) as f_in:
        for line in tqdm(list(f_in)):
            passage = json.loads(line)
            # tokenize document
            document_token = nltk.word_tokenize(passage['document']['text'])
            if 'title' in passage:
                title_token = nltk.word_tokenize(passage['title']['text'])
                passage['tokens'] = title_token + ['|'] + document_token
                # passage['tokens'] = title_token
            else:
                passage['tokens'] = document_token
            documents[int(passage['documentId'])] = passage
    return documents

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)

def char_vocab(word2id, data_path):
    # build char embeddings
    char_counter = Counter()
    max_char = 0
    with open(word2id) as f:
        for word in f:
            word = word.strip()
            max_char = max(max_char, len(word))
            for char in word:
                char_counter[char] += 1

    char2id = {c: idx for idx, c in enumerate(char_counter.keys(), 1)}
    char2id['__unk__'] = 0

    id2char = {id_:c for c, id_ in char2id.items()}

    vocab_size = len(char2id)
    char_vocabs = []
    for _ in range(vocab_size):
        char_vocabs.append(id2char[_])

    with open(data_path, 'w') as g:
        g.write('\n'.join(char_vocabs))

    print(max_char)

class LeftMMFixed(torch.autograd.Function):
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
    This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
    """

    def __init__(self):
        super(LeftMMFixed, self).__init__()
        self.sparse_weights = None

    def forward(self, sparse_weights, x):
        if self.sparse_weights is None:
            self.sparse_weights = sparse_weights
        return torch.mm(self.sparse_weights, x)

    def backward(self, grad_output):
        sparse_weights = self.sparse_weights
        return None, torch.mm(sparse_weights.t(), grad_output)


def sparse_bmm(X, Y):
    """Batch multiply X and Y where X is sparse, Y is dense.
    Args:
        X: Sparse tensor of size BxMxN. Consists of two tensors,
            I:3xZ indices, and V:1xZ values.
        Y: Dense tensor of size BxNxK.
    Returns:
        batched-matmul(X, Y): BxMxK
    """
    I = X._indices()
    V = X._values()
    B, M, N = X.size()
    _, _, K = Y.size()
    Z = I.size()[1]
    lookup = Y[I[0, :], I[2, :], :]
    X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
    S = use_cuda(Variable(torch.cuda.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
    prod_op = LeftMMFixed()
    prod = prod_op(S, lookup)
    return prod.view(B, M, K)

if __name__  == "__main__":
    # load_documents('datasets/wikimovie/full_doc/documents.json')
    char_vocab('datasets/webqsp/kb_05/vocab.txt', 'datasets/webqsp/kb_05/chars.txt')
