import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel

class Electra(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, details = {}) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.details = details
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["c2i"])
        self.article_class_num = len(maps["a2i"])
        self.hid_dim = hid_dim

        self.electra = AutoModel.from_pretrained("code/ptm/bert-base-chinese") 
        self.electra = AutoModel.from_pretrained("code/ptm/electra-small") 

        self.dropout = nn.Dropout(0.4)

        hid_dim = 256
        self.fc_ar1 = nn.Linear(hid_dim, hid_dim)
        self.fc_ch1 = nn.Linear(hid_dim, hid_dim)
        self.fc_pt1 = nn.Linear(hid_dim, hid_dim)
        self.fc_ar2 = nn.Linear(hid_dim, len(maps["a2i"]))
        self.fc_ch2 = nn.Linear(hid_dim, len(maps["c2i"]))
        self.fc_pt2 = nn.Linear(hid_dim, maps["pt_cls_len"])


    def enc(self, text):
        x = self.electra(text)
        out = x.last_hidden_state[:,0] # [256]
        return out
    
    def forward(self, data):
        fact_text = data["fact"]["input_ids"].cuda()
        fact_emb = self.enc(fact_text)

        out_ar = self.fc_ar2(nn.ReLU()(self.fc_ar1(fact_emb)))
        out_ch = self.fc_ch2(nn.ReLU()(self.fc_ch1(fact_emb)))
        out_pt = self.fc_pt2(nn.ReLU()(self.fc_pt1(fact_emb)))

        # out_ar = self.fc_ar2(fact_emb)
        # out_ch = self.fc_ch2(fact_emb)
        # out_pt = self.fc_pt2(fact_emb)

        return {
            "article": out_ar,
            "charge": out_ch,
            "penalty": out_pt,
            "cl_loss": torch.tensor(0.).cuda(),
            "meta": {}
        }
