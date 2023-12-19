import jieba
import re
import os
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
from tqdm import tqdm
import random
from gensim.models import Word2Vec
from utils.tokenizer import MyTokenizer
import numpy as np
import json
import scipy.sparse as sp
import scipy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RANDOM_SEED = 22
torch.manual_seed(RANDOM_SEED)


class CailDataset(Dataset):
    def __init__(self, dataset_name, facts, charges, articles, penaltys):
        self.facts = facts
        self.penaltys = torch.LongTensor(penaltys)

        if "single_label" in dataset_name:
            self.charges = torch.LongTensor(charges)
            self.articles = torch.LongTensor(articles)
        else:
            self.articles = torch.Tensor(articles)
            if "laic" in dataset_name:
                self.charges = torch.LongTensor(charges) # single
            else:
                self.charges = torch.Tensor(charges)

    def __getitem__(self, idx):
        return {
            "fact":
                {
                    "input_ids": self.facts["input_ids"][idx],
                    "token_type_ids": self.facts["token_type_ids"][idx],
                    "attention_mask": self.facts["attention_mask"][idx],
                },
            "article": self.articles[idx],
            "charge": self.charges[idx],
            "penalty": self.penaltys[idx],
        }

    def __len__(self):
        return len(self.articles)


class HANDataset(Dataset):
    def __init__(self, dataset_name, facts, charges, articles, penaltys):
        self.facts = facts
        self.penaltys = torch.LongTensor(penaltys)

        if "single_label" in dataset_name:
            self.charges = torch.LongTensor(charges)
            self.articles = torch.LongTensor(articles)
        else:
            self.articles = torch.Tensor(articles)
            if "laic" in dataset_name:
                self.charges = torch.LongTensor(charges) # single
            else:
                self.charges = torch.Tensor(charges)

    def __getitem__(self, idx):
        return {
            "fact":
                {
                    "input_ids": self.facts[idx]["input_ids"],
                    "doc_len": self.facts[idx]["doc_len"],
                    "sent_len": self.facts[idx]["sent_len"],
                },
            "article": self.articles[idx],
            "charge": self.charges[idx],
            "penalty": self.penaltys[idx],
        }

    def __len__(self):
        return len(self.articles)


# 标签转数字id
def label2idx(label, map):
    for i in range(len(label)):
        for j in range(len(label[i])):
            label[i][j] = map[str(label[i][j])]

    return label


def get_han_document(facts, tokenizer, max_doc_len = 15, max_sent_len = 100):
    def han_split_sent(fact):
        sents = fact.split("。")
        cur_doc_len = min(len(sents), max_doc_len)
        sents = tokenizer(sents, max_length=max_sent_len, return_tensors="pt", padding="max_length", truncation=True)
        cur_sent_len = sents["attention_mask"].sum(-1)
        sent_len_pad = torch.zeros((max_doc_len), dtype = torch.long)
        cur_sent_len = torch.cat((cur_sent_len, sent_len_pad), dim =0)
        cur_sent_len = cur_sent_len[:max_doc_len]

        doc_pad = torch.zeros((max_doc_len, max_sent_len), dtype=torch.long)
        sents = torch.cat((sents["input_ids"], doc_pad), dim = 0)
        sents = sents[:max_doc_len,:]
        return {
            "input_ids": sents,
            "doc_len": cur_doc_len,
            "sent_len": cur_sent_len,
        }
    ret_facts = []
    for fact in tqdm(facts):
        ret_facts.append(han_split_sent(fact))
    return ret_facts

def load_data(filepath, dataset_name, tokenizer):
    facts, articles, charges, penaltys = [], [], [], []
    with open(filepath) as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            ar = json_obj["meta"]["relevant_articles"]
            ch = json_obj["meta"]["accusation"]
            # pt = json_obj["meta"]["term_of_imprisonment"]["imprisonment"] # regression
            pt = json_obj["meta"]["pt_cls"] # classification

            if "single_label" in dataset_name:
                if len(ar) > 1 or len(ch) > 1: # single label
                    continue
            if "multi_label" in dataset_name:
                if len(ar) == 1 or len(ch) == 1: # multi label
                    continue

            facts.append(json_obj["fact"])
            articles.append(ar)
            charges.append(ch)
            penaltys.append(pt)

    pkl_path = "code/pkl/{}/train_clean.pkl".format(dataset_name)
    if not os.path.exists(pkl_path):
        path, _ = os.path.split(pkl_path)
        if not os.path.exists(path):
            os.makedirs(path)
        
        if "han" in dataset_name:
            facts = get_han_document(facts, tokenizer)
        else :
            newfacts = []
            for fact in facts:
                fact = fact.split(" ")
                if len(fact) > 510:
                    fact = fact[:255] + fact[-255:]
                fact = " ".join(fact)
                newfacts.append(fact)
            facts = tokenizer(newfacts, max_length=512, return_tensors="pt", padding="max_length", truncation=True)

        with open(pkl_path, "wb") as f:
            pickle.dump(facts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("pkl data saved: {}".format(pkl_path))
    with open(pkl_path, "rb") as f:
        facts = pickle.load(f)

    ret_maps = {}

    with open("data/filter/laic/meta/c2i.json") as f:
        c2i = json.load(f)
        ret_maps["c2i"] = c2i
        ret_maps["i2c"] = {v: k for k, v in c2i.items()}

    with open("data/filter/laic/meta/a2i.json") as f:
        a2i = json.load(f)
        ret_maps["a2i"] = a2i
        ret_maps["i2a"] = {v: k for k, v in a2i.items()}
    ret_maps["pt_cls_len"] = len(set(penaltys))
    charges = label2idx(charges, ret_maps["c2i"])
    articles = label2idx(articles, ret_maps["a2i"])

    def label_one_hot_pad(label, map):
        ret_label = np.zeros((len(label),len(map)))
        for i1 in range(len(label)):
            for i2 in label[i1]:
                ret_label[i1][i2] = 1
        return ret_label
    
    if "single_label" in dataset_name:
        articles = [a[0] for a in articles]
        charges = [c[0] for c in charges]
    else :
        articles = label_one_hot_pad(articles, ret_maps["a2i"])
        if "laic" in dataset_name:
            charges = [c[0] for c in charges]
        else:
            charges = label_one_hot_pad(charges, ret_maps["c2i"])

    if "han" in dataset_name:
        dataset = HANDataset(dataset_name, facts, charges, articles, penaltys)
    else:
        dataset = CailDataset(dataset_name, facts, charges, articles, penaltys)

    return dataset, ret_maps



class LaicNumDataset(Dataset):
    def __init__(self, dataset_name, facts, charges, articles, penaltys, mantissas):
        self.facts = facts
        self.penaltys = torch.LongTensor(penaltys)
        self.articles = torch.Tensor(articles)
        self.charges = torch.LongTensor(charges)
        self.mantissas = torch.LongTensor(mantissas)

    def __getitem__(self, idx):
        return {
            "fact":
                {
                    "input_ids": self.facts["input_ids"][idx],
                    "token_type_ids": self.facts["token_type_ids"][idx],
                    "attention_mask": self.facts["attention_mask"][idx],
                },
            "article": self.articles[idx],
            "charge": self.charges[idx],
            "penalty": self.penaltys[idx],
            "mantissa": self.mantissas[idx],
        }

    def __len__(self):
        return len(self.articles)


def load_data_num(filepath, dataset_name, tokenizer):
    facts, articles, charges, penaltys = [], [], [], []
    mant_data = []
    with open(filepath) as f:
        for line in tqdm(f.readlines()):
            json_obj = json.loads(line)
            ar = json_obj["meta"]["relevant_articles"]
            ch = json_obj["meta"]["accusation"]
            pt = json_obj["meta"]["pt_cls"] # classification
            fact = json_obj["fact"]

            fact = fact.split(" ")
            if len(fact) > 510:
                fact = fact[:255] + fact[-255:]
            fact = " ".join(fact)

            d_pattern = "\d+\.\d+|\d+"
            # fact = "蓄积 679 立方米 ， 造成 林 X 直接 经济损失 135710 元 。"
            nums = re.findall(d_pattern, fact)
            text = re.sub(d_pattern, " number ", fact)

            # nums_mant, nums_expo = [], []
            # for x in nums:
            #     x = float(x)
            #     if x < 1.0:
            #         mant, expo = x, "0"
            #     else:
            #         sci_num = format(float(x),'.2E')
            #         mant = float(sci_num.split("E+")[0])
            #         expo = str(int(sci_num.split("E+")[1]))
            #     nums_mant.append(mant)
            #     nums_expo.append(expo)

            # word_lst = text.split()
            # ptr = 0
            # mantissa_new = []
            # for i, w in enumerate(word_lst):
            #     if w == "number":
            #         if i+1 < len(word_lst) and word_lst[i+1] in ["元", "克","亩","株","棵"]:
            #             word_lst[i] = nums_expo[ptr]
            #             mantissa_new.append(nums_mant[ptr])
            #         ptr += 1
            # text = " ".join(word_lst)

            # mant_data.append(mantissa_new)
            facts.append(fact)
            articles.append(ar)
            charges.append(ch)
            penaltys.append(pt)

    pkl_path = "code/pkl/{}/train_clean.pkl".format(dataset_name)
    # if not os.path.exists(pkl_path):
        # path, _ = os.path.split(pkl_path)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        
    facts = tokenizer(facts, max_length=512, return_tensors="pt", padding="max_length", truncation=True)

        # with open(pkl_path, "wb") as f:
        #     pickle.dump(facts, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print("pkl data saved: {}".format(pkl_path))
    # with open(pkl_path, "rb") as f:
    #     facts = pickle.load(f)
    
    use_num_emb = False # 加速
    # use_num_emb = True
    if use_num_emb: 
        mantissa_lines = []
        for i in tqdm(range(len(charges))):
            cur_mant = mant_data[i]
            convert_text = tokenizer.convert_ids_to_tokens(facts["input_ids"][i])
            ptr = 0
            mantissa_line = [0] * len(convert_text)

            for i, w in enumerate(convert_text):
                if w in [str(x) for x in range(10)]:
                    mantissa_line[i] = cur_mant[ptr]
                    ptr += 1
            
            mantissa_lines.append(mantissa_line)
    else:
        mantissa_lines = [[0] * 512 for _ in range(len(charges))]


    ret_maps = {}

    with open("data/filter/laic/meta/c2i.json") as f:
        c2i = json.load(f)
        ret_maps["c2i"] = c2i
        ret_maps["i2c"] = {v: k for k, v in c2i.items()}

    with open("data/filter/laic/meta/a2i.json") as f:
        a2i = json.load(f)
        ret_maps["a2i"] = a2i
        ret_maps["i2a"] = {v: k for k, v in a2i.items()}
    ret_maps["pt_cls_len"] = len(set(penaltys))
    charges = label2idx(charges, ret_maps["c2i"])
    articles = label2idx(articles, ret_maps["a2i"])

    def label_one_hot_pad(label, map):
        ret_label = np.zeros((len(label),len(map)))
        for i1 in range(len(label)):
            for i2 in label[i1]:
                ret_label[i1][i2] = 1
        return ret_label
    
    articles = label_one_hot_pad(articles, ret_maps["a2i"])
    charges = [c[0] for c in charges]

    dataset = LaicNumDataset(dataset_name, facts, charges, articles, penaltys, mantissa_lines)

    return dataset, ret_maps


def load_details(filepath, cur_map, tokenizer, dataset_name, max_len = 200):
    details_dict = {}
    with open(filepath) as f:
        details_dict = json.load(f)
    if "charge" in filepath:
        # details_dict = {k: v["定义"] for k, v in details_dict.items()}
        details_dict = {k: v["定义"] + " " + v["客观方面"] for k, v in details_dict.items()}

    tmp_map = {k.replace("[","").replace("]","") : v for k, v in cur_map.items()}
    cur_map = tmp_map
    details = [""] * len(cur_map.keys())
    for label, text in details_dict.items():
        if label in cur_map.keys():
            details[cur_map[label]] = text
    
    if "han" in dataset_name:
        details = get_han_document(details, tokenizer, max_doc_len=5, max_sent_len=50)
        tmp_details = {}
        for key in details[0].keys():
            tmp_details[key] = torch.cat([torch.tensor(t[key]).unsqueeze(0).long() for t in details], dim = 0)
        details = tmp_details
    else :
        details = tokenizer(details, max_length = max_len, return_tensors = "pt", padding = "max_length", truncation = True)
    
    return details


def load_c2a(filepath = "data/filter/cail/meta/c2a_idx.json"):
    with open(filepath) as f:
        c2a = json.load(f)
    c2a = {int(k):int(v) for k,v in c2a.items()}
    return c2a


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.A

def load_article_adj(mode = "co"):
    def get_cail_data(filepath):
        articles, charges = [], []
        with open(filepath) as f:
            for line in f.readlines():
                json_obj = json.loads(line)
                ar = json_obj["meta"]["relevant_articles"]
                ch = json_obj["meta"]["accusation"]
                articles.append(list(set(ar)))
                charges.append(list(set(ch)))
        return articles, charges
    ars, chs = [], []
    filepath = "data/filter/laic/{}.json"

    for name in ["train","valid","test"]:
        fname = filepath.format(name)
        ar, ch = get_cail_data(fname)
        ars += ar
        chs += ch

    file_path = "data/filter/laic/meta/a2i.json"
    a2i = json.load(open(file_path))
    adj = [[0] * len(a2i)] * len(a2i)
    adj = np.array(adj)

    if mode == "co":
        for alst in ars:
            alst = set(alst)
            if len(alst) == 1:
                continue
            alst = [str(i) for i in alst]
            
            for a1 in alst:
                for a2 in alst:
                    if a1 == a2:
                        continue
                    adj[a2i[a1]][a2i[a2]] += 1

        adj += 100
        adj = np.log(adj)

    else:
        def load_details(filepath, cur_map):
            details_dict = {}
            details_dict = json.load(open(filepath))
            tmp_map = {k.replace("[","").replace("]","") : v for k, v in cur_map.items()}
            cur_map = tmp_map
            details = [""] * len(cur_map.keys())
            for label, text in details_dict.items():
                if label in cur_map.keys():
                    details[cur_map[label]] = text
            return details
        details = load_details("code/utils/article_details.json", a2i)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(details)
        vectorizer.get_feature_names_out()

        X = X.A
        adj = cosine_similarity(X, X)
    
    adj = scipy.sparse.csr_matrix(adj)
    adj = preprocess_adj(adj)
    return adj


def load_charge_adj():
    c2i = json.load(open("data/filter/laic/meta/c2i.json"))
    file_path = "code/utils/charge_details.json"
    def load_charge_details(filepath, cur_map):
        details_dict = {}
        details_dict = json.load(open(filepath))
        details_dict = {k: v["定义"] + " " + v["主观方面"] + " " + v["客观方面"] for k, v in details_dict.items()}
        tmp_map = {k.replace("[","").replace("]","") : v for k, v in cur_map.items()}
        cur_map = tmp_map
        details = [""] * len(cur_map.keys())
        for label, text in details_dict.items():
            if label in cur_map.keys():
                details[cur_map[label]] = text
        return details
    details = load_charge_details(file_path, c2i)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(details)
    vectorizer.get_feature_names_out()

    X = X.A
    adj = cosine_similarity(X, X)
    return preprocess_adj(adj)