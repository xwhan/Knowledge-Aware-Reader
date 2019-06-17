import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import AttnEncoder
from modules import Packed
from modules import SeqAttnMatch
from modules import l_relu
from modules import QueryReform
from modules import ConditionGate
from util import load_dict


class KAReader(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args):
        super(KAReader, self).__init__()

        self.entity2id = load_dict(args['data_folder'] + args['entity2id'])
        self.word2id = load_dict(args['data_folder'] + args['word2id'])
        self.relation2id = load_dict(args['data_folder'] + args['relation2id'])
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.num_word = len(self.word2id)
        self.num_layer = args['num_layer']
        self.use_doc = args['use_doc']
        self.word_drop = args['word_drop']
        self.hidden_drop = args['hidden_drop']
        self.label_smooth = args['label_smooth']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file'):
                setattr(self, k, args['data_folder'] + v)

        # pretrained entity embeddings
        self.entity_emb = nn.Embedding(self.num_entity + 1, self.entity_dim, padding_idx=self.num_entity)
        self.entity_emb.weight.data.copy_(torch.from_numpy(np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)), 'constant')))
        self.entity_emb.weight.requires_grad = False
        self.entity_linear = nn.Linear(self.entity_dim, self.entity_dim)

        # word embeddings
        self.word_emb = nn.Embedding(self.num_word, self.word_dim, padding_idx=1)
        self.word_emb.weight.data.copy_(torch.from_numpy(np.load(self.word_emb_file)))
        self.word_emb.weight.requires_grad = False

        self.word_emb_match = SeqAttnMatch(self.word_dim)

        self.hidden_dim = self.entity_dim
        # question and doc encoder
        self.question_encoder = Packed(nn.LSTM(self.word_dim, self.hidden_dim // 2, batch_first=True, bidirectional=True))

        # for shared encoder ablation
        # self.relation_encoder = Packed(nn.LSTM(self.word_dim, self.hidden_dim // 2, batch_first=True, bidirectional=True))

        self.self_att_r = AttnEncoder(self.hidden_dim)
        self.self_att_q = AttnEncoder(self.hidden_dim)
        self.combine_q_rel = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        # doc encoder

        self.ent_info_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_proj = nn.Linear(2*self.word_dim + 1, self.hidden_dim)
        self.doc_encoder = Packed(nn.LSTM(self.hidden_dim, self.hidden_dim // 2, batch_first=True, bidirectional=True))
        self.doc_to_ent = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ent_info_gate = ConditionGate(self.hidden_dim)
        self.ent_info_gate_out = ConditionGate(self.hidden_dim)

        self.kg_prop = nn.Linear(self.hidden_dim + self.entity_dim, self.entity_dim)
        self.kg_gate = nn.Linear(self.hidden_dim + self.entity_dim, self.entity_dim)
        self.self_prop = nn.Linear(self.entity_dim, self.entity_dim)
        self.combine_q = nn.Linear(2*self.hidden_dim, self.hidden_dim)

        self.reader_gate = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.query_update = QueryReform(self.hidden_dim)

        self.attn_match = nn.Linear(self.hidden_dim*3, self.hidden_dim*2)
        self.attn_match_q = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()

        self.word_drop = nn.Dropout(self.word_drop)
        self.hidden_drop = nn.Dropout(self.hidden_drop)

    def forward(self, feed):
        # encode questions
        question = feed['questions']
        q_mask = (question != 1).float()
        q_len = q_mask.sum(-1) # (B, q_len)
        q_word_emb = self.word_drop(self.word_emb(question))
        q_emb, _ = self.question_encoder(q_word_emb, q_len, max_length=question.size(1))
        q_emb = self.hidden_drop(q_emb)

        B, max_q_len = question.size(0), question.size(1)

        # candidate ent embeddings
        ent_emb_ = self.entity_emb(feed['candidate_entities'])
        ent_emb = l_relu(self.entity_linear(ent_emb_))

        # # keep a copy of the initial ent_emb
        # init_ent_emb = ent_emb
        ent_mask = (feed['candidate_entities'] != self.num_entity).float()

        # linked relations
        max_num_neighbors = feed['entity_link_ents'].size(2)
        max_num_candidates = feed['candidate_entities'].size(1)
        neighbor_mask = (feed['entity_link_ents'] != self.num_entity).float() # (B, |C|, |N|)

        # encode all relations with question encoder
        rel_word_ids = feed['rel_word_ids']
        rel_word_mask = (rel_word_ids != 1).float()
        rel_word_lens = rel_word_mask.sum(-1)
        rel_word_lens[rel_word_lens == 0] = 1
        rel_encoded, _ = self.question_encoder(self.word_drop(self.word_emb(rel_word_ids)), rel_word_lens, max_length=rel_word_ids.size(1)) # (|R|, r_len, h_dim)
        # rel_encoded, _ = self.relation_encoder(self.word_drop(self.word_emb(rel_word_ids)), rel_word_lens, max_length=rel_word_ids.size(1)) # (|R|, r_len, h_dim)
        rel_encoded = self.hidden_drop(rel_encoded)
        rel_encoded = self.self_att_r(rel_encoded, rel_word_mask)

        neighbor_rel_ids = feed['entity_link_rels'].long().view(-1)
        neighbor_rel_emb = torch.index_select(rel_encoded, dim=0, index=neighbor_rel_ids).view(B*max_num_candidates, max_num_neighbors, self.hidden_dim)

        # for look up
        neighbor_ent_local_index = feed['entity_link_ents'].long() # (B * |C| * max_num_neighbors)
        neighbor_ent_local_index = neighbor_ent_local_index.view(B, -1)
        neighbor_ent_local_mask = (neighbor_ent_local_index != -1).long()
        fix_index = torch.arange(B).long() * max_num_candidates
        fix_index = fix_index.to(torch.device('cuda'))
        neighbor_ent_local_index = neighbor_ent_local_index + fix_index.view(-1,1)
        neighbor_ent_local_index = (neighbor_ent_local_index + 1) * neighbor_ent_local_mask
        neighbor_ent_local_index = neighbor_ent_local_index.view(-1)

        # prepare pagerank scores
        ent_seed_info = feed['query_entities'].float() # seed entity will have 1.0 score
        ent_pagerank = torch.cat([torch.zeros(1).to(torch.device('cuda')), ent_seed_info.view(-1)], dim=0)
        pagerank = torch.index_select(ent_pagerank, dim=0, index=neighbor_ent_local_index).view(B*max_num_candidates, max_num_neighbors)

        # v0.0 more find-grained attention
        q_emb_expand = q_emb.unsqueeze(1).expand(B, max_num_candidates, max_q_len, -1).contiguous()
        q_emb_expand = q_emb_expand.view(B*max_num_candidates, max_q_len, -1)
        q_mask_expand = q_mask.unsqueeze(1).expand(B, max_num_candidates, -1).contiguous()
        q_mask_expand = q_mask_expand.view(B*max_num_candidates, -1)
        q_n_affinity = torch.bmm(q_emb_expand, neighbor_rel_emb.transpose(1, 2)) # (bsize*max_num_candidates, q_len, max_num_neighbors)
        q_n_affinity_mask_q = q_n_affinity - (1 - q_mask_expand.unsqueeze(2)) * 1e8
        q_n_affinity_mask_n = q_n_affinity - (1 - neighbor_mask.view(B*max_num_candidates, 1, max_num_neighbors))
        normalize_over_q = F.softmax(q_n_affinity_mask_q, dim=1)
        normalize_over_n = F.softmax(q_n_affinity_mask_n, dim=2)
        retrieve_q = torch.bmm(normalize_over_q.transpose(1,2), q_emb_expand)
        q_rel_simi = torch.sum(neighbor_rel_emb * retrieve_q, dim=2)

        init_q_emb = self.self_att_r(q_emb, q_mask)

        retrieve_r = torch.bmm(normalize_over_n, neighbor_rel_emb)
        q_and_rel = torch.cat([q_emb_expand, retrieve_r], dim=2)
        rel_aware_q = self.combine_q_rel(q_and_rel).tanh().view(B, max_num_candidates, -1, self.hidden_dim)

        # pooling over the q_len dim
        q_node_emb = rel_aware_q.max(2)[0]

        ent_emb = l_relu(self.combine_q(torch.cat([ent_emb, q_node_emb], dim=2)))
        ent_emb_for_lookup = ent_emb.view(-1, self.entity_dim)
        ent_emb_for_lookup = torch.cat([torch.zeros(1, self.entity_dim).to(torch.device('cuda')), ent_emb_for_lookup], dim=0)
        neighbor_ent_emb = torch.index_select(ent_emb_for_lookup, dim=0, index=neighbor_ent_local_index)
        neighbor_ent_emb = neighbor_ent_emb.view(B*max_num_candidates, max_num_neighbors, -1)
        neighbor_vec = torch.cat([neighbor_rel_emb, neighbor_ent_emb], dim =-1).view(B*max_num_candidates, max_num_neighbors, -1) # for propagation
        neighbor_scores = q_rel_simi * pagerank
        neighbor_scores = neighbor_scores - (1 - neighbor_mask.view(B*max_num_candidates, max_num_neighbors)) * 1e8
        attn_score = F.softmax(neighbor_scores, dim=1)
        aggregate = self.kg_prop(neighbor_vec) * attn_score.unsqueeze(2)
        aggregate = l_relu(aggregate.sum(1)).view(B, max_num_candidates, -1)
        self_prop_ = l_relu(self.self_prop(ent_emb))
        gate_value = self.kg_gate(torch.cat([aggregate, ent_emb], dim = -1)).sigmoid()
        ent_emb = gate_value * self_prop_ + (1 - gate_value) * aggregate

        # read documents
        if self.use_doc:
            q_for_text = self.query_update(init_q_emb, ent_emb, ent_seed_info, ent_mask)
            # q_for_text = q_node_emb.mean(1)
            # q_for_text = init_q_emb

            q_node_emb = torch.cat([q_node_emb, q_for_text.unsqueeze(1).expand_as(q_node_emb).contiguous()], dim=-1) 

            ent_linked_doc_spans = feed['ent_link_doc_spans']
            doc = feed['documents'] # (B, |D|, d_len)
            max_num_doc = doc.size(1)
            max_d_len = doc.size(2)
            doc_mask = (doc != 1).float()
            doc_len = doc_mask.sum(-1)
            doc_len += (doc_len == 0).float() # padded documents have 0 words
            doc_len = doc_len.view(-1)
            d_word_emb = self.word_drop(self.word_emb(doc.view(-1, doc.size(-1)))) # (B*|D|, d_len, emb_dim)
            
            # input features for documents
            q_word_emb = q_word_emb.unsqueeze(1).expand(B, max_num_doc, max_q_len, self.word_dim).contiguous()
            q_word_emb = q_word_emb.view(B*max_num_doc, max_q_len, -1)
            q_mask_ = (question == 1).unsqueeze(1).expand(B, max_num_doc, max_q_len).contiguous()
            q_mask_ = q_mask_.view(B*max_num_doc, -1)
            q_weighted_emb = self.word_emb_match(d_word_emb, q_word_emb, q_mask_)
            doc_em = feed['documents_em'].float().view(B*max_num_doc, max_d_len, 1)
            doc_input = torch.cat([d_word_emb, q_weighted_emb, doc_em], dim=-1) # 2*word_dim + 1

            doc_input = self.input_proj(doc_input).tanh()
            word_entity_id = ent_linked_doc_spans.view(B, max_num_candidates, -1).transpose(1,2)
            word_ent_info_mask = (word_entity_id.sum(-1, keepdim=True) != 0).float()
            word_ent_info = torch.bmm(word_entity_id.float(), ent_emb) # (B, |D|*d_len, h_dim)
            word_ent_info = self.ent_info_proj(word_ent_info).tanh()
            doc_input = self.ent_info_gate(q_for_text.unsqueeze(1), word_ent_info, doc_input.view(B, max_num_doc*max_d_len, -1), word_ent_info_mask)

            d_emb, _ = self.doc_encoder(doc_input.view(B*max_num_doc, max_d_len, -1), doc_len, max_length=doc.size(2))
            d_emb = self.hidden_drop(d_emb)

            d_emb = self.ent_info_gate_out(q_for_text.unsqueeze(1), word_ent_info, d_emb.view(B, max_num_doc*max_d_len, -1), word_ent_info_mask).view(B*max_num_doc, max_d_len, -1)

            q_for_text = q_for_text.unsqueeze(1).expand(B, max_num_doc, self.hidden_dim).contiguous()
            q_for_text = q_for_text.view(B*max_num_doc, -1) # (B*|D|, h_dim)
            d_emb = d_emb.view(B*max_num_doc, max_d_len, -1) # (B*|D|, d_len, h_dim)
            q_over_d = torch.bmm(q_for_text.unsqueeze(1), d_emb.transpose(1,2)).squeeze(1) # (B*|D|, d_len)
            q_over_d = F.softmax(q_over_d - (1 - doc_mask.view(B*max_num_doc, max_d_len))*1e8, dim=-1)
            q_retrieve_d = torch.bmm(q_over_d.unsqueeze(1), d_emb).view(B, max_num_doc, -1) # (B, |D|, h_dim)
            ent_linked_doc = (ent_linked_doc_spans.sum(-1) != 0).float() # (B, |C|, |D|)
            ent_emb_from_doc = torch.bmm(ent_linked_doc, q_retrieve_d) # (B, |C|, h_dim)
            # ent_emb_from_doc = F.dropout(ent_emb_from_doc, 0.5, self.training)

            # retrieve_span
            ent_emb_from_span = torch.bmm(feed['ent_link_doc_norm_spans'].float().view(B, max_num_candidates, -1), d_emb.view(B, max_num_doc*max_d_len, -1))
            ent_emb_from_span = F.dropout(ent_emb_from_span, 0.2, self.training)


        # refine KB ent_emb
        # refined_ent_emb = self.refine_ent(ent_emb, ent_emb_from_doc)
        if self.use_doc:
            ent_emb = self.attn_match(torch.cat([ent_emb, ent_emb_from_doc, ent_emb_from_span], dim=-1)).relu()
            # q_node_emb = self.attn_match_q(q_node_emb).tanh()

        ent_scores = (q_node_emb * ent_emb).sum(2)

        answers = feed['answers'].float()
        if self.label_smooth:
            answers = ((1.0 - self.label_smooth)*answers) + (self.label_smooth/answers.size(1))

        loss = self.loss(ent_scores, feed['answers'].float())

        pred_dist = (ent_scores - (1-ent_mask) * 1e8).sigmoid() * ent_mask
        pred = torch.max(ent_scores, dim=1)[1]

        return loss, pred, pred_dist
