import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

dataset = 'datasets/webqsp/kb_05'
rel_path = dataset + '/relations.txt'

word_counter = []

# load original vocab
with open(dataset + '/vocab.txt') as f:
    for line in f.readlines():
        word_counter.append(line.strip())

rel_words = []
max_num_words = 0
all_relations = []

# how to split the relation
if 'webqsp' in dataset:
    with open(rel_path) as f:
        first_line = True
        for line in tqdm(f.readlines()):
            if first_line:
                first_line = False
                continue
            line = line.strip()
            all_relations.append(line)
            line = line[1:-1]
            fields = line.split('.')
            words = fields[-2].split('_') + fields[-1].split('_')
            max_num_words = max(len(words), max_num_words)
            rel_words.append(words)
            word_counter += words
elif 'wikimovie' in dataset:
    with open(rel_path) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            all_relations.append(line)
            words = line.split('_')
            max_num_words = max(len(words), max_num_words)
            rel_words.append(words)
            word_counter += words

print('max_num_words: ', max_num_words)

word_counter = nlp.data.count_tokens(word_counter)
glove_emb = nlp.embedding.create('glove', source='glove.6B.100d')
vocab = nlp.Vocab(word_counter)
vocab.set_embedding(glove_emb)

emb_mat = vocab.embedding.idx_to_vec.asnumpy()
np.save(dataset + '/glove_word_emb_100d', emb_mat)

with open(dataset + '/glove_vocab.txt', 'w') as g:
    g.write('\n'.join(vocab.idx_to_token))

assert False

rel_word_ids = np.ones((len(rel_words) + 1, max_num_words), dtype=int) # leave the first 1 for padding relation
rel_emb_mat = []
for rel_idx, words in enumerate(rel_words):
    for i, word in enumerate(words):
        rel_word_ids[rel_idx + 1, i] = vocab.token_to_idx[word]

np.save(dataset + '/rel_word_idx', rel_word_ids)

all_relations = ['pad_rel'] + all_relations
with open(rel_path, 'w') as g:
    g.write('\n'.join(all_relations))



