from .ToksTransformer import *
from .data_utils import *
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .GAT import GAT
from .SupConLoss import SupConLoss

DROPOUT = 0.3


class ELELJP_Num(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, details={}):
        super().__init__()
        self.ptm_path = "code/ptm/electra-small"
        config = AutoConfig.from_pretrained(self.ptm_path)       
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT    
        self.bert = AutoModel.from_pretrained(self.ptm_path)

        self.ar_cls = len(maps["a2i"])
        self.ch_cls = len(maps["c2i"])
        self.pt_cls = maps["pt_cls_len"]

        self.article_det_transformer = DetTransformer(vocab_size, num_hidden_layers = 3)
        self.charge_det_transformer = DetTransformer(vocab_size, num_hidden_layers = 3)

        hid_dim = 256 if "small" in self.ptm_path else 768
        self.hid_dim = hid_dim
        
        self.fc_ch1 = torch.nn.Linear(2 * hid_dim, hid_dim)
        self.fc_ar1 = torch.nn.Linear(2 * hid_dim, hid_dim)

        self.fc_ar2 = torch.nn.Linear(hid_dim, self.ar_cls)
        self.fc_ch2 = torch.nn.Linear(hid_dim, self.ch_cls)

        self.fc_pt1 = torch.nn.Linear(3 * hid_dim, hid_dim)
        self.fc_pt2 = torch.nn.Linear(hid_dim, self.pt_cls)

        self.tanh = nn.Tanh()
        self.details = details
        self.maps = maps

        self.ca_role_embedding = nn.Parameter(torch.zeros(hid_dim))
        self.pa_role_embedding = nn.Parameter(torch.zeros(hid_dim))

        self.GAT_ar = GAT(nfeat=hid_dim, outfeat = self.ar_cls, maps = maps, dropout=0.2, nheads=8)
        # self.GAT_ch = GAT(nfeat=hid_dim, outfeat = self.accu_cls, maps = maps, dropout=0.4, nheads=8)
        self.sup_con_loss = SupConLoss(scale_by_temperature = False)

    def get_penalty_gat(self, label_specific_emb, mask = None):
        return self.GAT_ar(label_specific_emb, self.ar_adj, mask)

    def get_mantissa_embedding(self, mantissa_emb):
        q_uniform = torch.linspace(-10, 10, self.hid_dim).cuda()
        q_uniform = q_uniform.expand(mantissa_emb.shape)
        NE = torch.exp(-(mantissa_emb - q_uniform)**2 * 0.025)
        NE = torch.where(mantissa_emb>0, NE,  torch.zeros_like(NE).float())
        return NE

    def text_enc(self, data):
        text = data["fact"]
        if self.hid_dim == 256:
            self.hid_dim //= 2

        input_ids, token_type_ids, attention_mask = text["input_ids"].cuda(), text["token_type_ids"].cuda(), text["attention_mask"].cuda()
        embeddings = self.bert.embeddings(input_ids)

        mantissa_emb = data["mantissa"].cuda().unsqueeze(-1).repeat([1,1,self.hid_dim])
        mantissa_embedding = self.get_mantissa_embedding(mantissa_emb)
        exponent_embeddings = torch.where(mantissa_emb>0, embeddings, torch.zeros_like(embeddings).float())
        unit_embeddings = embeddings.clone()
        unit_embeddings[:,:-1,:] = unit_embeddings[:,1:,:].clone()
        unit_embeddings = torch.where(mantissa_emb>0, unit_embeddings, torch.zeros_like(embeddings).float())
        number_embeddings = mantissa_embedding * 0.2 + exponent_embeddings * 0.4 + unit_embeddings * 0.4

        # 添加数值编码
        embeddings = torch.where(mantissa_emb > 0, number_embeddings, embeddings)

        if "small" in self.ptm_path:
            embeddings = self.bert.embeddings_project(embeddings)

        # encoder_input = embeddings
        output = self.bert.encoder(embeddings)
        return output.last_hidden_state

    def get_mask_adj(self, label):
        batch, label_num = label.shape
        adj = label.unsqueeze(-1)
        adj = adj.expand(batch, label_num, label_num)
        adj2 = adj.transpose(1, 2)
        adj = adj + adj2
        ones = torch.ones_like(adj)
        adj = torch.where(adj > 1.5, ones, -ones)
        return adj 

    def get_pred01(self, pred):
        threshold = 0
        zero = torch.zeros_like(pred)
        one = torch.ones_like(pred)
        pred = torch.where(pred <= threshold, zero, pred)
        pred = torch.where(pred > threshold, one, pred)
        return pred
    
    def forward(self, data):
        fact_emb = self.text_enc(data)
        article_text = self.details["a_details"]
        charge_text = self.details["c_details"]
        af_emb, a_emb = self.article_det_transformer(hidden_states=fact_emb, det_text = article_text)
        cf_emb, c_emb = self.charge_det_transformer(hidden_states=fact_emb, det_text = charge_text)
        af_emb = self.tanh(af_emb)
        cf_emb = self.tanh(cf_emb)

        # ar_pred = self.ar_pred(af_emb).squeeze(-1) # [4, 70]
        # ch_pred = self.ch_pred(cf_emb).squeeze(-1)

        gt_ar, gt_ch = data["article"].cuda(), data["charge"].cuda()

        fact_pool_emb = torch.max(fact_emb, dim = 1)[0].unsqueeze(1)
        # 罪名预测
        cf_pool_emb = torch.max(cf_emb, dim = 1)[0].unsqueeze(1)
        ch_emb = torch.cat([fact_pool_emb, cf_pool_emb], dim = 1).view(len(fact_pool_emb),-1)
        ch_pred = self.fc_ch2(nn.ReLU()(self.fc_ch1(ch_emb)))

        # 法条预测
        af_pool_emb = torch.max(af_emb, dim = 1)[0].unsqueeze(1)
        ar_emb = torch.cat([fact_pool_emb, af_pool_emb], dim = 1).view(len(fact_pool_emb),-1)
        ar_pred = self.fc_ar2(nn.ReLU()(self.fc_ar1(ar_emb)))
        
        # 添加role embedding
        pa_role_embs = self.pa_role_embedding.expand(af_emb.shape)
        ca_role_embs = self.ca_role_embedding.expand(af_emb.shape)
        role_embs = torch.cat([pa_role_embs[:,:12,:], ca_role_embs[:,12:,:]], dim = 1)
        # af_emb = af_emb + role_embs

        # GAT
        # pred_ar = self.get_pred01(ar_pred)
        # gt_ar = pred_ar
        ar_adj = self.get_mask_adj(gt_ar)
        af_emb_tilde = self.GAT_ar(af_emb, ar_adj)
        af_pool_emb_tilde = torch.max(af_emb_tilde, dim = 1)[0].unsqueeze(1)
        pt_emb = torch.cat([fact_pool_emb, af_pool_emb_tilde, cf_pool_emb], dim = 1).view(len(fact_pool_emb),-1)

        pt_pred = self.fc_pt2(nn.ReLU()(self.fc_pt1(pt_emb))) 

        return {
            "article": ar_pred,
            "charge": ch_pred,
            "penalty": pt_pred,
            "cl_emb": {
                "af_emb": af_emb,
                "a_emb": a_emb,
                "cf_emb": cf_emb,
                "c_emb": c_emb,
            },
            "meta": {
                "af_emb": af_emb,
                "a_emb": a_emb,
                "cf_emb": cf_emb,
                "c_emb": c_emb,
            }
        }

