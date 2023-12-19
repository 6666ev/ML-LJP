import copy
import math
import torch
import numpy as np
import pickle as pkl
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

law_cls = 103
accu_cls = 119
term_cls = 11

# law_cls = 70
# accu_cls = 42
# term_cls = 9
hidden_size = 256


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class GraphConvolution(nn.Module):

    def __init__(self, opt, adj):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        m = torch.matmul(x, self.weight)
        m = torch.matmul(self.adj, m)
        return m


class GNN(nn.Module):
    def __init__(self, num_feature, num_class, adj):
        super(GNN, self).__init__()
        self.adj = adj
        hidden_dim = 16
        opt_ = dict([('in', num_feature), ('out', hidden_dim)])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', hidden_dim), ('out', num_class)])
        self.m2 = GraphConvolution(opt_, adj)

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = self.m1(x)
        x = F.relu(x)
        x = self.m2(x)
        x = x.squeeze(-1)
        # return x[:, :law_cls], x[:, law_cls:law_cls + accu_cls], x[:, law_cls + accu_cls:law_cls + accu_cls + term_cls]
        return x

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)


class ToksIntraDistillation(nn.Module):
    def __init__(self):
        super(ToksIntraDistillation, self).__init__()
        self.s0 = nn.Linear(hidden_size, hidden_size)
        self.s1 = nn.Linear(hidden_size, hidden_size)
        self.s2 = nn.Linear(hidden_size, hidden_size)
        self.s3 = nn.Linear(hidden_size, hidden_size)

    def opt(self, toks):
        s0 = self.s0(toks)
        s1 = self.s1(toks)
        s2 = self.s2(toks)
        s3 = self.s3(toks)
        mask = torch.eye(toks.shape[1]) * -10000.0
        mask = mask.cuda()
        similarity_scores = torch.matmul(s1, s2.transpose(-1, -2)) / math.sqrt(hidden_size) + mask
        similarity_probs = nn.Softmax(dim=-1)(similarity_scores)
        return s0 - torch.matmul(similarity_probs, s3).contiguous()

    def forward(self, toks):
        law_toks = self.opt(toks[:, :law_cls, :])
        accu_toks = self.opt(toks[:, law_cls:law_cls + accu_cls, :])
        term_toks = self.opt(toks[:, law_cls + accu_cls:law_cls + accu_cls + term_cls, :])
        return torch.cat([law_toks, accu_toks, term_toks], dim=1).contiguous()


class ToksAttention(nn.Module):
    def __init__(self, strategy=0, multihead=False):
        super(ToksAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.num_attention_heads = 8
        if multihead:
            self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 12*64=768
        else:
            self.attention_head_size = hidden_size
        self.all_head_size = hidden_size
        self.multihead = multihead
        law2accu = pkl.load(open('data/rformer/law2accu.pkl', 'rb'))
        law2term = pkl.load(open('data/rformer/law2term.pkl', 'rb'))
        accu2term = pkl.load(open("data/rformer/accu2term.pkl", "rb"))
        if strategy == 0:
            '''
            law_acc:84.8,
            '''
            mask = torch.ones((law_cls + accu_cls + term_cls, law_cls + accu_cls + term_cls))
        elif strategy == 1:
            # 没有监督的二部图
            mask = torch.eye(law_cls + accu_cls + term_cls)
            mask[:law_cls, law_cls:law_cls + accu_cls] = torch.ones(
                (law_cls, accu_cls))  # torch.from_numpy(law2accu).long()
            mask[:law_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = torch.ones(
                (law_cls, term_cls))  # torch.from_numpy(law2term).long()
            mask[:, :law_cls] = mask[:law_cls].T
            mask[law_cls:law_cls + accu_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = torch.ones(
                (accu_cls, term_cls))
            mask[law_cls + accu_cls:law_cls + accu_cls + term_cls, law_cls:law_cls + accu_cls] = torch.ones(
                (term_cls, accu_cls))
        elif strategy == 2:
            # 没有监督也没A-T的二部图
            mask = torch.eye(law_cls + accu_cls + term_cls)
            mask[:law_cls, law_cls:law_cls + accu_cls] = torch.ones(
                (law_cls, accu_cls))  # torch.from_numpy(law2accu).long()
            mask[:law_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = torch.ones(
                (law_cls, term_cls))  # torch.from_numpy(law2term).long()
            mask[:, :law_cls] = mask[:law_cls].T
        elif strategy == 3:
            # 有监督但没A-T的二部图
            '''
            law_acc:84.3
            '''
            mask = torch.eye(law_cls + accu_cls + term_cls)
            mask[:law_cls, law_cls:law_cls + accu_cls] = torch.from_numpy(law2accu).long()
            mask[:law_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = torch.from_numpy(law2term).long()
            mask[:, :law_cls] = mask[:law_cls].T
        elif strategy == 4:
            # 有监督的二部图
            mask = torch.eye(law_cls + accu_cls + term_cls)
            mask[:law_cls, law_cls:law_cls + accu_cls] = torch.from_numpy(law2accu).long()
            mask[:law_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = torch.from_numpy(law2term).long()
            mask[:, :law_cls] = mask[:law_cls].T
            mask[law_cls:law_cls + accu_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = torch.from_numpy(
                accu2term).long()
            mask[law_cls + accu_cls:law_cls + accu_cls + term_cls, law_cls:law_cls + accu_cls] = torch.from_numpy(
                accu2term).T.long()
        mask = (1.0 - mask) * -10000.0
        self.mask = mask.cuda()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, toks):
        if self.multihead:
            mixed_query_layer = self.query(toks)
            mixed_key_layer = self.key(toks)
            mixed_value_layer = self.value(toks)
            # law2accu,accu2law,law2term,term2law
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
        else:
            query_layer = self.query(toks)
            key_layer = self.key(toks)
            value_layer = self.value(toks)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + self.mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if self.multihead:
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.contiguous()
        return context_layer


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings.
    """

    def __init__(self):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(law_cls + accu_cls + term_cls, 768, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(3, 768)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertCrossOutput(nn.Module):
    def __init__(self):
        super(BertCrossOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ToksInterAttention(nn.Module):
    def __init__(self, s, m=True):
        super(ToksInterAttention, self).__init__()
        # self.cross = ToksAttention(strategy=s, multihead=m)
        self.cross = nn.MultiheadAttention(hidden_size, num_heads=12, batch_first=True)
        self.output = BertCrossOutput()

    def forward(self, toks):
        # cross_output = self.cross(toks)
        cross_output, _ = self.cross(toks, toks, toks)
        attention_output = self.output(cross_output, toks)
        return attention_output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        intermediate_size = 3072
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        intermediate_size = 3072
        hidden_dropout_prob = 0.1
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, s=0, m=True):
        super(BertLayer, self).__init__()
        self.attention = ToksInterAttention(s=s, m=m)
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, toks):
        attention_output = self.attention(toks)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLayerWithDistill(nn.Module):
    def __init__(self, s=0, m=True):
        super(BertLayerWithDistill, self).__init__()
        self.attention = ToksInterAttention(s=s, m=m)
        self.intermediate1 = BertIntermediate()
        self.distillation = ToksIntraDistillation()
        self.intermediate2 = BertIntermediate()
        self.output1 = BertOutput()
        self.output2 = BertOutput()

    def forward(self, toks):
        attention_output = self.attention(toks)
        intermediate_output = self.intermediate1(attention_output)
        layer_output = self.output1(intermediate_output, attention_output)

        distillation_output = self.distillation(layer_output)
        intermediate_output = self.intermediate2(distillation_output)
        layer_output = self.output2(intermediate_output, distillation_output)
        return layer_output


class ToksTransformer(nn.Module):
    def __init__(self, cls_num):
        super(ToksTransformer, self).__init__()
        self.embeddings = DetEmbeddings(cls_num)
        self.cls_num = cls_num
        self.input_ids = torch.range(0, cls_num - 1).unsqueeze(0).long().cuda()
        self.mh_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

    def forward(self, hidden_states):
        det_emb = self.embeddings(self.input_ids)
        det_emb = det_emb.repeat((len(hidden_states), 1, 1))
        hn, _ = self.mh_attn(det_emb, hidden_states, hidden_states)
        return hn, det_emb

class DetEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings.
    """
    def __init__(self, vocab_size):
        super(DetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DetTransformer(nn.Module):
    def __init__(self, vocab_size, num_hidden_layers = 3):
        super(DetTransformer, self).__init__()
        self.embeddings = DetEmbeddings(vocab_size)
        self.layer = nn.ModuleList([nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True, dropout=0.3) for _ in range(num_hidden_layers)])
        self.mh_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.pooling = "max_pooling"
        self.pooling = "cls"

    def forward(self, hidden_states, det_text):
        batch = hidden_states.shape[0]
        input_ids, token_type_ids, attention_mask = det_text["input_ids"].cuda(), det_text["token_type_ids"].cuda(), det_text["attention_mask"].cuda().bool()

        det_num = len(input_ids)
        det_emb = self.embeddings(input_ids)

        for layer_module in self.layer:
            det_emb = layer_module(det_emb)
        det_emb = det_emb.permute(0, 2, 1)
        det_emb = F.max_pool1d(det_emb, det_emb.size(2)).squeeze(2)

        # if self.pooling == "max_pooling":
        #     det_emb = F.max_pool1d(det_emb, det_emb.size(2)).squeeze(2)
        # elif self.pooling == "cls":
        #     det_emb = det_emb[:,:,0]
        det_emb = det_emb.unsqueeze(0)
        det_emb = det_emb.expand([batch, det_num, hidden_size]) # [batch, 70, dim]
        hn, _ = self.mh_attn(det_emb, hidden_states, hidden_states) # [batch, seq len, dim]
        return hn, det_emb
