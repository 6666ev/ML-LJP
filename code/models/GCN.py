import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 bias=False,
                 activation=F.relu):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs
        x = F.dropout(x, self.dropout) # [batch, cls_num, dim]

        xw = torch.matmul(x, self.weight)

        batch = x.shape[0]
        cur_sup = support.unsqueeze(0).repeat((batch, 1, 1)).float()
        out = torch.bmm(cur_sup, xw)

        out += self.bias
        return self.activation(out), support


class GCNet(nn.Module):
    def __init__(self, input_dim, hid_dim, cls_num, adj):
        super(GCNet, self).__init__()

        self.input_dim = input_dim  # input_dim
        self.hid_dim = hid_dim 
        self.cls_num = cls_num  # 7
        self.adj = adj
        self.fc = nn.Linear(self.adj.shape[0], 1)
        self.dropout = 0.5

        self.layers = nn.Sequential(
            GraphConvolution(
                self.input_dim,  # 256
                self.hid_dim,  # 256
                activation=F.relu,
                dropout=self.dropout),
            GraphConvolution(
                self.hid_dim,  # 256
                self.hid_dim,  # 256
                activation=F.relu,
                dropout=self.dropout),
            GraphConvolution(
                self.hid_dim,  # 256
                self.hid_dim,  # 256
                activation=F.relu,
                dropout=self.dropout),
            GraphConvolution(
                self.hid_dim,  # 256
                self.hid_dim,  # 256
                activation=F.relu,
                dropout=self.dropout),
            GraphConvolution(
                self.hid_dim,  # 256
                self.hid_dim,  # 256
                activation=F.relu,
                dropout=self.dropout),
            GraphConvolution(
                self.hid_dim, # 256
                cls_num, # 70
                activation=F.relu,
                dropout=self.dropout),
            )

    def forward(self, inputs):
        x, support = inputs
        x, _ = self.layers((x, support))
        x = x.transpose(1,2)
        logits = self.fc(x)
        return logits.squeeze(-1)

