import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, nfeat, outfeat, dropout, nheads, maps = None, alpha = 0.2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        nhid = int(nfeat / nheads)
        self.attention_layers = nn.Sequential(
            GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha=alpha),
            GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha=alpha),
            GraphAttentionLayer(nfeat, nhid, nheads, dropout, alpha=alpha),
            )

        # self.out_att = GraphAttentionHead(
        #     len(maps["a2i"]), maps["pt_cls_len"], dropout=dropout, alpha=alpha, activate_function=None)
        # self.fc = nn.Linear(outfeat, maps["pt_cls_len"])

    def forward(self, x, adj):
        x, _ = self.attention_layers((x, adj))
        return x
        x = x.transpose(1,2)
        x = self.fc(x).squeeze(-1)
        return x.transpose(1,2)


class GraphAttentionLayer(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout, alpha = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GraphAttentionHead(
            nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        # for i, attention in enumerate(self.attentions):
            # self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionHead(
            nfeat, nfeat, dropout=dropout, alpha=alpha)

    def forward(self, inputs):
        x, adj = inputs # x: [batch, ar_num, dim]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1) # x: [batch, ar_num, dim]
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x, adj


class GraphAttentionHead(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha = 0.2, activate_function = F.elu):
        super(GraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.activate_function = activate_function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # Wh = torch.mm(h, self.W)
        Wh = torch.matmul(h, self.W.unsqueeze(0))
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -1e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.activate_function is not None:
            return self.activate_function(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1,2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
