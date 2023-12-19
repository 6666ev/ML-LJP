from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from glob import glob
import pickle as pkl
import numpy as np
import random
import torch
import json
import re


def send_email(content):
    import smtplib
    from email.mime.text import MIMEText
    msg_from = ''
    passwd = ''
    msg_to = ''
    subject = "log信息"
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        # s.sendmail(msg_from, "284467290@qq.com", msg.as_string())
    except:
        pass


def gen_mp():
    dataset = CrimeFactDataset("train")
    law_mp_accu = {}
    law_mp_term = {}
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        if sample['law'] not in law_mp_accu:
            law_mp_accu[sample['law']] = [0] * 117
        law_mp_accu[sample['law']][sample['accu']] += 1
        if sample['law'] not in law_mp_term:
            law_mp_term[sample['law']] = [0] * 11
        law_mp_term[sample['law']][sample['term']] += 1
    pkl.dump(law_mp_term, open("law_mp_term.pkl", "wb"))
    pkl.dump(law_mp_accu, open("law_mp_term.pkl", "wb"))


def collate_fn_law(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    label = torch.stack([x['label'] for x in batch]).squeeze()
    return input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), masked_lm_labels.cuda(), label.cuda().long()


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    label = [x['label'] for x in batch]
    label = torch.from_numpy(np.array(label))
    start = [x['start'] for x in batch]
    start = torch.from_numpy(np.array(start))
    end = [x['end'] for x in batch]
    end = torch.from_numpy(np.array(end))
    return input_ids.cuda(), \
           token_type_ids.cuda(), \
           attention_mask.cuda(), \
           masked_lm_labels.cuda(), \
           label.cuda().long(), \
           start.cuda().long(), \
           end.cuda().long()


def collate_fn_fact_pkl(batch):
    rep = torch.stack([torch.from_numpy(x['rep']) for x in batch]).squeeze(1)
    emb = torch.stack([torch.from_numpy(x['emb']) for x in batch]).squeeze(1)
    mask = torch.stack([torch.from_numpy(x['mask']) for x in batch]).squeeze(1)
    accu = [x['accu'] for x in batch]
    accu = torch.from_numpy(np.array(accu))
    law = [x['law'] for x in batch]
    law = torch.from_numpy(np.array(law))
    term = [x['term'] for x in batch]
    term = torch.from_numpy(np.array(term))
    return rep.cuda(), emb.cuda(), mask.cuda(), law.cuda().long(), accu.cuda().long(), term.cuda().long()


def collate_fn_fact(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    accu = [x['accu'] for x in batch]
    accu = torch.from_numpy(np.array(accu))
    law = [x['law'] for x in batch]
    law = torch.from_numpy(np.array(law))
    term = [x['term'] for x in batch]
    term = torch.from_numpy(np.array(term))
    return input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), \
           law.cuda().long(), accu.cuda().long(), term.cuda().long(), \
           masked_lm_labels.cuda()


def collate_fn_gnn_p(batch):
    inputs = [x['inputs'] for x in batch]
    inputs = torch.from_numpy(np.array(inputs))
    targets = [x['targets'] for x in batch]
    targets = torch.from_numpy(np.array(targets))
    return inputs.unsqueeze(-1).cuda(), targets.cuda()


class CrimeFactDatasetPKL(Dataset):
    def __init__(self, mode):
        self.fact_list = glob("../dataset/%s_cs/*.pkl" % mode)

    def __getitem__(self, item):
        return pkl.load(open(self.fact_list[item], "rb"))

    def __len__(self):
        return len(self.fact_list)



class CrimeFactDataset(Dataset):
    def __init__(self, mode, mask_rate=0.):
        fact_list = open("data/laic/%s.json" % mode, 'r', encoding='UTF-8').readlines()
        delete_index = []
        for i, sample in enumerate(fact_list):
            sample = json.loads(sample)
            if len(sample['fact']) <= 10:
                delete_index.append(i)
        delete_index = delete_index[::-1]
        for idx in delete_index:
            fact_list.pop(idx)
        self.fact_list = fact_list
        self.tokenizer = BertTokenizer.from_pretrained("code/ptm/bert-base-chinese")
        self.max_length = 512
        self.mask_rate = mask_rate

        self.c2i = {}
        with open("new_accu.txt") as f:
            for i, c in enumerate(f.readlines()):
                c = c.strip()
                self.c2i[c] = i
        self.a2i = {}
        with open("new_law.txt") as f:
            for i, a in enumerate(f.readlines()):
                a = a.strip()
                self.a2i[a] = i
        
        self.c2i = json.load(open("data/laic/meta/c2i.json"))
        self.a2i = json.load(open("data/laic/meta/a2i.json"))

    def __getitem__(self, item):
        sample = self.fact_list[item]
        sample = json.loads(sample)
        fact = sample['fact'].replace(" ", "")
        if len(fact) > 510:
            fact = fact[:255] + fact[-255:]
        ret = self.tokenizer(fact, max_length=512, padding="max_length", return_tensors="pt")
        fact_tok = self.tokenizer.tokenize(fact)
        ret['masked_lm_labels'] = ret['input_ids'].clone()
        length = int(torch.sum(ret['attention_mask']))
        ret['masked_lm_labels'][0][0] = ret['masked_lm_labels'][0][length - 1] = -1
        ret['input_ids'][0][random.choices(range(1, length), k=int((length - 2) * self.mask_rate))] = 103
        ret['accu'] = sample["meta"]['accusation'][0]
        ret['accu'] = self.c2i[ret['accu']]
        ret['law'] = max(sample["meta"]['relevant_articles'])
        ret['law'] = self.a2i[str(ret['law'])]
        ret['term'] = sample["meta"]['pt_cls']
        ret['fact_tok'] = fact_tok
        return ret

    def __len__(self):
        return len(self.fact_list)

