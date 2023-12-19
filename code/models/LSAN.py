import torch
import torch.nn.functional as F


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path


class LSAN(BasicModule):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, details={}):
        super(LSAN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.maps = maps
        n_classes = len(maps["a2i"])
        self.n_classes = n_classes
        self.embeddings = self._load_embeddings(vocab_size, hid_dim)
        self.label_embed = self.load_labelembedd(n_classes, hid_dim)

        self.lstm = torch.nn.LSTM(hid_dim, hidden_size=hid_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(hid_dim * 2, hid_dim)
        self.linear_second = torch.nn.Linear(hid_dim, n_classes)

        self.weight1 = torch.nn.Linear(hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(hid_dim * 2, 1)

        self.output_layer = torch.nn.Linear(hid_dim*2, n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.3)

    def _load_embeddings(self, vocab_size, hid_dim):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(vocab_size, hid_dim)
        # word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings
    
    def load_labelembedd(self, label_num, hid_dim):
        """Load the embeddings based on flag"""
        embed = torch.nn.Embedding(label_num, hid_dim)
        # embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def init_hidden(self, batch_size):
        return (torch.randn(2,batch_size,self.hid_dim).cuda(),torch.randn(2,batch_size,self.hid_dim).cuda())

    def forward(self,data):
        x = data["fact"]["input_ids"].cuda()
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)
        #step1 get LSTM outputs
        batch_size = x.shape[0]
        hidden_state = self.init_hidden(batch_size)
        outputs, hidden_state = self.lstm(embeddings, hidden_state)
        #step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt= selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)   
        #step3 get label-attention
        h1 = outputs[:, :, :self.hid_dim]
        h2 = outputs[:, :,self.hid_dim:]
        
        label = self.label_embed.weight.data
        m1 = torch.bmm(label.expand(batch_size, self.n_classes, self.hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(batch_size, self.n_classes, self.hid_dim), h2.transpose(1, 2))
        label_att= torch.cat((torch.bmm(m1,h1),torch.bmm(m2,h2)),2)
        # label_att = F.normalize(label_att, p=2, dim=-1)
        # self_att = F.normalize(self_att, p=2, dim=-1) #all can
        weight1=torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att ))
        weight1 = weight1/(weight1+weight2)
        weight2= 1-weight1

        doc = weight1*label_att+weight2*self_att
        # there two method, for simple, just add
        # also can use linear to do it
        avg_sentence_embeddings = torch.sum(doc, 1)/self.n_classes

        # pred = torch.sigmoid(self.output_layer(avg_sentence_embeddings))
        out_ar = self.output_layer(avg_sentence_embeddings)

        return {
            "article": out_ar,
            "meta": {}
        }