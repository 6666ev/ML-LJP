import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
from collections import OrderedDict

from utils.tokenizer import MyTokenizer


cx = None

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()

        self.emb_dim = 300
        self.output_dim = self.emb_dim // 4

        self.min_gram = 2
        self.max_gram = 5
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, self.output_dim, (a, self.emb_dim)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = self.emb_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.view(batch_size, 1, -1, self.emb_dim)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        return conv_out

class LSTMEncoder(nn.Module):
    def __init__(self):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = 300
        self.bi = True
        self.output_size = self.hidden_size
        self.num_layers = 3
        if self.bi:
            self.output_size = self.output_size // 2

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bi)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        # print(x.size())
        # print(batch_size, self.num_layers + int(self.bi) * self.num_layers, self.output_size)
        hidden = (
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).cuda(),
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).cuda())

        h, c = self.lstm(x, hidden)

        h_ = torch.max(h, dim=1)[0]

        return h_, h

class LJPPredictor(nn.Module):
    def __init__(self, maps):
        super(LJPPredictor, self).__init__()

        self.hidden_size = 300
        self.charge_fc = nn.Linear(self.hidden_size, len(maps["c2i"]))
        self.article_fc = nn.Linear(self.hidden_size, len(maps["a2i"]))
        self.term_fc = nn.Linear(self.hidden_size, maps["pt_cls_len"])

    def forward(self, h):
        charge = self.charge_fc(h)
        article = self.article_fc(h)
        term = self.term_fc(h)
        return {"zm": charge, "ft": article, "xq": term}

class GatingLayer(nn.Module):
    def __init__(self):
        super(GatingLayer, self).__init__()

        num_layers = 3
        # config.set("model", "num_layers", 1)
        self.encoder = LSTMEncoder()
        # config.set("model", "num_layers", num_layers)

        self.hidden_size = 300
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.15)


    def forward(self, h):
        _, h = self.encoder(h)
        g = self.fc(torch.cat([h, cx], dim=2))
        h = g * h
        h = self.dropout(h)
        return h


class CPTP(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, details={}) -> None:
        super(CPTP, self).__init__()
        self.encoder = []
        num_layers = 3
        for a in range(0, num_layers):
            self.encoder.append(("Layer%d" % a, GatingLayer()))

        self.encoder = nn.Sequential(OrderedDict(self.encoder))
        self.fc = LJPPredictor(maps)

        self.hidden_size = 300
        self.tokenizer = MyTokenizer(
            embedding_path="code/gensim_train/word2vec.model")
        vectors = self.tokenizer.load_embedding()
        vectors = torch.Tensor(vectors)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(vectors)


        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        # self.word_embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))), self.hidden_size)
        self.charge_embedding = nn.Embedding(202, self.hidden_size)

        self.fake_tensor = []
        for a in range(0, 202):
            self.fake_tensor.append(a)
        self.fake_tensor = Variable(torch.LongTensor(self.fake_tensor)).cuda()
        self.cnn = CNNEncoder()
        self.dropout = nn.Dropout(0.15)


    def forward(self, data):
        x = data["fact"]["input_ids"].cuda()
        x = self.word_embedding(x)


        batch_size = x.size()[0]
        c = self.fake_tensor.view(1, -1).repeat(batch_size, 1)
        c = self.charge_embedding(c)
        zm = data["charge"].view(batch_size, -1, 1).repeat(1, 1, self.hidden_size)
        zm = zm.cuda()
        c = c * zm.float()
        c = torch.max(c, dim=1)[0]

        global cx
        cx = c.view(batch_size, 1, -1).repeat(1,  512, 1)
        y = self.encoder(x)
        y = self.dropout(y)
        y = self.cnn(y)
        y = self.dropout(y)

        result = self.fc(y)
        
        return {
            "article": result["ft"],
            "charge": result["zm"],
            "penalty": result["xq"],
            "cl_loss": torch.tensor(0).cuda(),
            "meta": {}
        }


        loss = 0
        for name in ["zm", "ft", "xq"]:
            loss += self.criterion[name](result[name], data[name])

        if acc_result is None:
            acc_result = {"zm": None, "ft": None, "xq": None}

        for name in ["zm", "ft", "xq"]:
            acc_result[name] = self.accuracy_function[name](result[name], data[name], config, acc_result[name])

        return {"loss": loss, "acc_result": acc_result}


