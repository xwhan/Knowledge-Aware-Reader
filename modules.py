import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Packed(nn.Module):

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    @property
    def batch_first(self):
        return self.rnn.batch_first

    def forward(self, inputs, lengths, hidden=None, max_length=None):
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices]
        outputs, (h, c) = self.rnn(nn.utils.rnn.pack_padded_sequence(inputs, lens.tolist(), batch_first=self.batch_first), hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first, total_length=max_length)
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def l_relu(x, n_slope=0.01):
    return F.leaky_relu(x, n_slope)

class ConditionGate(nn.Module):
    """docstring for ConditionGate"""
    def __init__(self, h_dim):
        super(ConditionGate, self).__init__()
        self.gate = nn.Linear(2*h_dim, h_dim, bias=False)
        # self.q_to_x = nn.Linear(h_dim, h_dim)
        # self.q_to_y = nn.Linear(h_dim, h_dim)
        
    def forward(self, q, x, y, gate_mask):
        q_x_sim = x*q
        q_y_sim = y*q
        gate_val = self.gate(torch.cat([q_x_sim, q_y_sim], dim=-1)).sigmoid()
        gate_val = gate_val * gate_mask
        return gate_val * x  + (1 - gate_val) * y


class Fusion(nn.Module):
    """docstring for Fusion"""
    def __init__(self, d_hid):
        super(Fusion, self).__init__()
        self.r = nn.Linear(d_hid*4, d_hid, bias=False)
        self.g = nn.Linear(d_hid*4, d_hid, bias=False)

    def forward(self, x, y):
        r_ = self.r(torch.cat([x,y,x-y,x*y], dim=-1)).tanh()
        g_ = torch.sigmoid(self.g(torch.cat([x,y,x-y,x*y], dim=-1)))
        return g_ * r_ + (1 - g_) * x

class AttnEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, d_hid):
        super(AttnEncoder, self).__init__()
        self.attn_linear = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x, x_mask):
        """
        x: (B, len, d_hid)
        x_mask: (B, len)
        return: (B, d_hid)
        """
        x_attn = self.attn_linear(x)
        x_attn = x_attn - (1 - x_mask.unsqueeze(2))*1e8
        x_attn = F.softmax(x_attn, dim=1)
        return (x*x_attn).sum(1)

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                # ...Otherwise 0-1 probabilities
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha

class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)

        return matched_seq


class QueryReform(nn.Module):
    """docstring for QueryReform"""
    def __init__(self, h_dim):
        super(QueryReform, self).__init__()
        # self.q_encoder = AttnEncoder(h_dim)
        self.fusion = Fusion(h_dim)
        self.q_ent_attn = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, ent_emb, seed_info, ent_mask):
        '''
        q: (B,q_len,h_dim)
        q_mask: (B,q_len)
        q_ent_span: (B,q_len)
        ent_emb: (B,C,h_dim)
        seed_info: (B, C)
        ent_mask: (B, C)
        '''
        # q_node = self.q_encoder(q, q_mask)
        q_ent_attn = (self.q_ent_attn(q_node).unsqueeze(1) * ent_emb).sum(2, keepdim=True)
        q_ent_attn = F.softmax(q_ent_attn - (1 - ent_mask.unsqueeze(2)) * 1e8, dim=1)
        # attn_retrieve = (q_ent_attn * ent_emb).sum(1)

        seed_retrieve = torch.bmm(seed_info.unsqueeze(1), ent_emb).squeeze(1) # (B, 1, h_dim)
        # how to calculate the gate

        # return  self.fusion(q_node, attn_retrieve)
        return  self.fusion(q_node, seed_retrieve)


        # retrieved = self.transform(torch.cat([seed_retrieve, attn_retrieve], dim=-1)).relu()
        # gate_val = self.gate(torch.cat([q.squeeze(1), seed_retrieve, attn_retrieve], dim=-1)).sigmoid()
        # return self.fusion(q.squeeze(1), retrieved).unsqueeze(1)
        # return (gate_val * q.squeeze(1) + (1 - gate_val) * torch.tanh(self.transform(torch.cat([q.squeeze(1), seed_retrieve, attn_retrieve], dim=-1)))).unsqueeze(1)

