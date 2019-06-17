import json
import nltk
import numpy as np
import random
import torch

from collections import defaultdict
from tqdm import tqdm
from util import get_config
from util import load_dict
from util import load_documents

class DataLoader():
    def __init__(self, config, documents, mode='train'):
        self.mode = mode
        self.use_doc = config['use_doc']
        self.use_inverse_relation = config['use_inverse_relation']
        self.max_query_word = config['max_query_word']
        self.max_document_word = config['max_document_word']
        self.max_char = config['max_char']
        self.documents = documents
        self.data_file = config['data_folder'] + config['{}_data'.format(mode)]
        self.batch_size = config['batch_size'] if mode == 'train' else config['batch_size']
        self.max_rel_words = config['max_rel_words']
        self.type_rels = config['type_rels']
        self.fact_drop = config['fact_drop']

        # read all data
        self.data = []
        with open(self.data_file) as f:
            for line in tqdm(list(f)):
                self.data.append(json.loads(line))

        # word and kb vocab
        self.word2id = load_dict(config['data_folder'] + config['word2id'])
        self.relation2id = load_dict(config['data_folder'] + config['relation2id'])
        self.entity2id = load_dict(config['data_folder'] + config['entity2id'])
        self.id2entity = {i:entity for entity, i in self.entity2id.items()}

        self.rel_word_idx = np.load(config['data_folder'] + 'rel_word_idx.npy')

        # for batching
        self.max_local_entity = 0 # max num of candidates
        self.max_relevant_docs = 0 # max num of retired documents
        self.max_kb_neighbors = config['max_num_neighbors'] # max num of neighbors for entity
        self.max_kb_neighbors_ = config['max_num_neighbors'] # kb relations are directed
        self.max_linked_entities = 0 # max num of linked entities for each doc
        self.max_linked_documents = 50 # max num of linked documents for each entity

        self.num_kb_relation = 2 * len(self.relation2id) if self.use_inverse_relation  else len(self.relation2id)

        # get the batching parameters
        self.get_stats()

    def get_stats(self):
        if self.use_doc:
            # max_linked_entities
            self.useful_docs = {} # filter out documents with out linked entities
            for docid, doc in self.documents.items():
                linked_entities = 0
                if 'title' in doc:
                    linked_entities += len(doc['title']['entities'])
                    offset = len(nltk.word_tokenize(doc['title']['text']))
                else:
                    offset = 0
                for ent in doc['document']['entities']:
                    if ent['start'] + offset >= self.max_document_word:
                        continue
                    else:
                        linked_entities += 1
                if linked_entities > 1:
                    self.useful_docs[docid] = doc
                self.max_linked_entities = max(self.max_linked_entities, linked_entities)
            print('max num of linked entities: ', self.max_linked_entities)

        # decide how many neighbors should we consider
        # num_neighbors = []

        num_tuples = []
        
        # max_linked_documents, max_relevant_docs, max_local_entity
        for line in tqdm(self.data):
            candidate_ents = set()
            rel_docs = 0

            # question entity
            for ent in line['entities']:
                candidate_ents.add(ent['text'])
            # kb entities
            for ent in line['subgraph']['entities']:
                candidate_ents.add(ent['text'])

            num_tuples.append(line['subgraph']['tuples'])

            if self.use_doc:
                # entities in doc
                for passage in line['passages']:
                    if passage['document_id'] not in self.useful_docs:
                        continue
                    rel_docs += 1
                    document = self.useful_docs[int(passage['document_id'])]
                    for ent in document['document']['entities']:
                        candidate_ents.add(ent['text'])
                    if 'title' in document:
                        for ent in document['title']['entities']:
                            candidate_ents.add(ent['text'])

            neighbors = defaultdict(list)
            neighbors_ = defaultdict(list)

            for triple in line['subgraph']['tuples']:
                s, r, o = triple
                neighbors[s['text']].append((r['text'], o['text']))
                neighbors_[o['text']].append((r['text'], s['text']))

            self.max_relevant_docs = max(self.max_relevant_docs, rel_docs)
            self.max_local_entity = max(self.max_local_entity, len(candidate_ents))

        # np.save('num_neighbors_', num_neighbors)

        print('mean num of triples: ', len(num_tuples))

        print('max num of relevant docs: ', self.max_relevant_docs)
        print('max num of candidate entities: ', self.max_local_entity)
        print('max_num of neighbors: ', self.max_kb_neighbors)
        print('max_num of neighbors inverse: ', self.max_kb_neighbors_)

    def batcher(self, shuffle=False):
        if shuffle:
            random.shuffle(self.data)

        device = torch.device('cuda')

        for batch_id in tqdm(range(0, len(self.data), self.batch_size)):
            batch = self.data[batch_id:batch_id + self.batch_size]

            batch_size = len(batch)
            questions = np.full((batch_size, self.max_query_word), 1, dtype=int)
            documents = np.full((batch_size, self.max_relevant_docs, self.max_document_word), 1, dtype=int)
            entity_link_documents = np.zeros((batch_size, self.max_local_entity, self.max_linked_documents, self.max_document_word), dtype=int)
            entity_link_doc_norm = np.zeros((batch_size, self.max_local_entity, self.max_linked_documents, self.max_document_word), dtype=int)
            documents_ans_span = np.zeros((batch_size, self.max_relevant_docs, 2), dtype=int)
            entity_link_ents = np.full((batch_size, self.max_local_entity, self.max_kb_neighbors_), -1, dtype=int) # incoming edges
            entity_link_rels = np.zeros((batch_size, self.max_local_entity, self.max_kb_neighbors_), dtype=int)
            candidate_entities = np.full((batch_size, self.max_local_entity), len(self.entity2id), dtype=int)
            ent_degrees = np.zeros((batch_size, self.max_local_entity), dtype=int)
            true_answers = np.zeros((batch_size, self.max_local_entity), dtype=float)
            query_entities = np.zeros((batch_size, self.max_local_entity), dtype=float)
            answers_ = []
            questions_ = []
            
            for i, sample in enumerate(batch):
                doc_global2local = {}
                # answer set
                answers = set()
                for answer in sample['answers']:
                    keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                    answers.add(self.entity2id[answer[keyword]])

                if self.mode != 'train':
                    answers_.append(list(answers))
                    questions_.append(sample['question'])
                
                # candidate entities, linked_documents
                candidates = set()
                query_entity = set()
                ent2linked_docId = defaultdict(list)
                for ent in sample['entities']:
                    candidates.add(self.entity2id[ent['text']])
                    query_entity.add(self.entity2id[ent['text']])
                for ent in sample['subgraph']['entities']:
                    candidates.add(self.entity2id[ent['text']])

                if self.use_doc:
                    for local_id, passage in enumerate(sample['passages']):
                        if passage['document_id'] not in self.useful_docs:
                            continue
                        doc_id = int(passage['document_id'])
                        doc_global2local[doc_id] = local_id
                        document = self.useful_docs[doc_id]
                        for word_pos, word in enumerate(['<bos>'] + document['tokens']):
                            if word_pos < self.max_document_word:
                                documents[i, local_id, word_pos] = self.word2id.get(word, self.word2id['<unk>'])
                        for ent in document['document']['entities']:
                            if self.entity2id[ent['text']] in answers:
                                documents_ans_span[i, local_id, 0] = min(ent['start'] + 1, self.max_document_word-1)
                                documents_ans_span[i, local_id, 1] = min(ent['end'] + 1, self.max_document_word-1)
                            s, e = ent['start'] + 1, ent['end'] + 1
                            ent2linked_docId[self.entity2id[ent['text']]].append((doc_id, s, e))
                            candidates.add(self.entity2id[ent['text']])
                        if 'title' in document:
                            for ent in document['title']['entities']:
                                candidates.add(self.entity2id(ent['text']))

                # kb information
                connections = defaultdict(list)

                if self.fact_drop and self.mode == 'train':
                    all_triples = sample['subgraph']['tuples']
                    random.shuffle(all_triples)
                    num_triples = len(all_triples)
                    keep_ratio = 1 - self.fact_drop
                    all_triples = all_triples[:int(num_triples * keep_ratio)]

                else:
                    all_triples = sample['subgraph']['tuples']

                for tpl in all_triples:
                    s,r,o = tpl


                    # only consider one direction of information propagation
                    connections[self.entity2id[o['text']]].append((self.relation2id[r['text']], self.entity2id[s['text']]))

                    if r['text'] in self.type_rels:
                        connections[self.entity2id[s['text']]].append((self.relation2id[r['text']], self.entity2id[o['text']]))


                # used for updating entity representations
                ent_global2local = {}
                candidates = list(candidates)

                # if len(candidates) == 0:
                    # print('No entities????')
                    # print(sample)

                for j, entid in enumerate(candidates):
                    if entid in query_entity:
                        query_entities[i, j] = 1.0
                    candidate_entities[i, j] = entid
                    ent_global2local[entid] = j
                    if entid in answers: true_answers[i, j] = 1.0
                    for linked_doc in ent2linked_docId[entid]:
                        start, end = linked_doc[1], linked_doc[2]
                        if end - start > 0:
                            entity_link_documents[i, j, doc_global2local[linked_doc[0]], start:end] = 1.0
                            entity_link_doc_norm[i, j, doc_global2local[linked_doc[0]], start:end] = 1.0 

                for j, entid in enumerate(candidates):
                    for count, neighbor in enumerate(connections[entid]):
                        if count < self.max_kb_neighbors_:
                            r_id, s_id = neighbor
                            # convert the global ent id to subgraph id, for graph convolution
                            s_id_local = ent_global2local[s_id]
                            entity_link_rels[i, j, count] = r_id
                            entity_link_ents[i, j, count] = s_id_local
                            ent_degrees[i, s_id_local] += 1

                # questions
                for j, word in enumerate(sample['question'].split()):
                    if j < self.max_query_word:
                        if word in self.word2id:
                            questions[i, j] = self.word2id[word]
                        else: 
                            questions[i, j] = self.word2id['<unk>']

            if self.use_doc:
                # exact match features for docs
                d_cat = documents.reshape((batch_size, -1))
                em_d = np.array([np.isin(d_, q_) for d_, q_ in zip(d_cat, questions)], dtype=int) # exact match features
                em_d = em_d.reshape((batch_size, self.max_relevant_docs, -1))

            batch_dict = {
                'questions': questions, # (B, q_len)
                'candidate_entities': candidate_entities,
                'entity_link_ents': entity_link_ents,
                'answers': true_answers,
                'query_entities': query_entities,
                'answers_': answers_,
                'questions_': questions_,
                'rel_word_ids': self.rel_word_idx, # (num_rel+1, word_lens)
                'entity_link_rels': entity_link_rels, # (bsize, max_num_candidates, max_num_neighbors)
                'ent_degrees': ent_degrees
            }

            if self.use_doc:
                batch_dict['documents'] = documents
                batch_dict['documents_em'] = em_d
                batch_dict['ent_link_doc_spans'] = entity_link_documents
                batch_dict['documents_ans_span'] = documents_ans_span
                batch_dict['ent_link_doc_norm_spans'] = entity_link_doc_norm

            for k, v in batch_dict.items():
                if k.endswith('_'):
                    batch_dict[k] = v
                    continue
                if not self.use_doc and 'doc' in k:
                    continue
                batch_dict[k] = torch.from_numpy(v).to(device)
            yield batch_dict


if __name__ == '__main__':
    cfg = get_config()
    documents = load_documents(cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
    # cfg['batch_size'] = 2
    train_data = DataLoader(cfg, documents)
    # build_squad_like_data(cfg['data_folder'] + cfg['{}_data'.format(cfg['mode'])], cfg['data_folder'] + cfg['{}_documents'.format(cfg['mode'])])
    for batch in train_data.batcher():
        print(batch['documents_ans_span'])
        assert False