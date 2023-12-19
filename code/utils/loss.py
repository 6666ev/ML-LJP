import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from models.SupConLoss import SupConLoss
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

def log_square_loss(y_pred, y_true):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    return torch.mean((torch.log(torch.clamp(y_pred, 0, 450) + 1) - torch.log(torch.clamp(y_true, 0, 450) + 1)) ** 2)


def log_sum_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.concat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.concat([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)

    return torch.mean(neg_loss + pos_loss)


def log_dis(y_true, y_pred):
    # 128：batch size
    # 450应该是最大刑期37年，将y_preds限幅到0~37年
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    return float(torch.mean(torch.log(torch.abs(torch.clamp(y_pred, 0, 450) - torch.clamp(y_true, 0, 450)) + 1)))


def acc25(y_true, y_pred):
    y_pred = y_pred.squeeze()
    y_pred = y_pred.round()
    y_true = y_true.squeeze()
    return int(torch.sum(torch.abs(y_pred-y_true)/y_true < 0.25))/len(y_pred)

def log_dis_np(y_true, y_pred):
    return float(np.mean(np.log(np.abs(np.clip(y_pred, 0, 450) - np.clip(y_true, 0, 450)) + 1)))


def acc25_np(y_true, y_pred):
    return int(np.sum(np.abs(y_pred-y_true)/y_true < 0.25))/len(y_pred)

sup_con_loss = SupConLoss()

def get_contrastive_loss_sup(lf_emb, label = None):
    label_idx = label.view(-1).cpu().numpy()
    if len(label.shape) == 1:
        single_label = label
    else :
        args = np.argwhere(label.cpu().numpy())
        single_label = torch.tensor(args[:,1]).cuda()
        label_idx = np.argwhere(label_idx == 1)
    lf_emb = lf_emb.reshape(-1, lf_emb.shape[-1])
    lf_emb = lf_emb[label_idx.reshape(-1), :]

    return sup_con_loss(lf_emb, labels = single_label)

def get_contrastive_loss_unsup(lf_emb, l_emb, label):
    tot_label_emb = l_emb[0] # [70, d]
    label_idx = label.view(-1).cpu().numpy()
    if len(label.shape) == 1:
        single_label = label
    else :
        args = np.argwhere(label.cpu().numpy())
        single_label = torch.tensor(args[:,1]).cuda()
        label_idx = np.argwhere(label_idx == 1)
        
    lf_emb = lf_emb.reshape(-1, lf_emb.shape[-1]) # [b, 70, d]
    lf_emb = lf_emb[label_idx.reshape(-1), :] # [p, d]

    l_emb = l_emb.reshape(-1, l_emb.shape[-1]) # [b, 70, d]
    l_emb = l_emb[label_idx.reshape(-1), :] # [p, d]

    cos_pos = nn.CosineSimilarity()(lf_emb, l_emb)
    lf_emb = nn.functional.normalize(lf_emb, dim=-1)
    tot_label_emb = nn.functional.normalize(tot_label_emb, dim=-1)
    sum_term = torch.mm(lf_emb, tot_label_emb.transpose(0,1)).squeeze(-1)  # [p, 70]

    temprature = 0.07
    pos_term = -torch.log(torch.exp(cos_pos / temprature))
    sum_term = torch.logsumexp(sum_term / temprature, dim = -1)
    loss = pos_term + sum_term
    return loss.mean()

def get_contrastive_loss_simcse(af_emb, af_emb_aug, label):
    label_idx = label.view(-1).cpu().numpy()
    if len(label.shape) == 1:
        single_label = label
    else :
        args = np.argwhere(label.cpu().numpy())
        single_label = torch.tensor(args[:,1]).cuda()
        label_idx = np.argwhere(label_idx == 1)
        
    af_emb = af_emb.reshape(-1, af_emb.shape[-1])
    af_emb = af_emb[label_idx.reshape(-1), :]
    af_emb_aug = af_emb_aug.reshape(-1, af_emb_aug.shape[-1])
    af_emb_aug = af_emb_aug[label_idx.reshape(-1), :]

    tot = torch.zeros([2 * len(af_emb), af_emb.shape[-1]]).cuda()
    idx = torch.arange(len(af_emb))
    tot[idx * 2, :] = af_emb
    tot[idx * 2 + 1, :] = af_emb_aug
    return simcse_unsup_loss(tot)

def simcse_unsup_loss(y_pred):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0]).cuda()
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0]).cuda() * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

def dthreshold_loss(y_true, y_pred, threshold):
    if len(threshold) == 140:
        threshold = (threshold[:70] + threshold[70:])/2
    margin = y_pred - threshold
    seg = y_true * 2 - 1
    margin = margin * seg
    loss = torch.log(torch.exp(-margin) + 1)
    # loss = loss.sum(-1)
    penalty = (threshold * threshold).sum()
    return loss.mean() + penalty

def a_cos_loss(a_emb):
    a_cos = a_emb
    a_cos = nn.functional.normalize(a_cos, dim=-1)
    return torch.mm(a_cos, a_cos.T).mean()

def cal_cls_f1(y_true, pred, threshold):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)
    pred = torch.where(pred <= threshold, zero, pred)
    pred = torch.where(pred > threshold, one, pred)
    y_pred = pred.numpy()
    res = classification_report(y_true, y_pred, output_dict=True)
    return [[res[str(cls)]['f1-score'], res[str(cls)]['support']]  for cls in range(y_true.shape[-1])]
    return [res[str(cls)]['f1-score'] + res[str(cls)]['precision'] + res[str(cls)]['recall']  for cls in range(y_true.shape[-1])]


def update_threshold(y_true, y_pred):
    cls_num = y_true.shape[-1]
    threshold = []
    search_net = [[] for _ in range(cls_num)]
    for ti in tqdm(np.linspace(-1, 1, 41)):
        f1_res = cal_cls_f1(y_true, y_pred, ti)
        for c in range(cls_num):
            search_net[c].append([ti] + f1_res[c])
    for lst in search_net:
        lst.sort(key = lambda x: x[1], reverse=True)
        threshold.append(lst[0][0])
    return torch.tensor(threshold)
