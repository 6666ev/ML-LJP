from turtle import update
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from utils.tokenizer import MyTokenizer, THULAC_Tokenizer
import jieba
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn import metrics
import json
import numpy as np
import os
from models import *
from utils import loader
import argparse
import time
from utils.loss import * 
import warnings
from torch.utils.tensorboard import SummaryWriter
import sys

warnings.filterwarnings('ignore')


RANDOM_SEED = 2022

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(RANDOM_SEED)

name2model = {
    "HAN": HAN,
    "FLA": FLA,
    "CPTP": CPTP,
    "LSAN": LSAN,

    "Electra": Electra,
    "ELELJP_Num": ELELJP_Num,
    "HAN_BERT": HAN_BERT,
}

name2tokenizer = {
    "HAN": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),
    "FLA": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),
    "CPTP": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),
    "LSAN": MyTokenizer(embedding_path="code/gensim_train/word2vec.model"),

    "Electra": AutoTokenizer.from_pretrained("code/ptm/electra-base"),
    "ELELJP_Num": AutoTokenizer.from_pretrained("code/ptm/electra-base"),
    "HAN_BERT": AutoTokenizer.from_pretrained("code/ptm/electra-base"),
}

name2dim = {
    "HAN": 300,
    "FLA": 300,
    "CPTP": 300,
    "LSAN": 300,

    "Electra": -1,
    "ELELJP_Num": -1,
    "HAN_BERT": -1
}


class Trainer:
    def __init__(self, args):

        self.tokenizer = name2tokenizer[args.model_name]
        self.dataset_name = args.data_name

        dataset_name = os.path.join(args.data_name,"train")
        data_path = "data/{}.json".format(dataset_name)
        print("当前数据集路径: ", data_path)
        self.trainset, self.maps = loader.load_data_num(data_path, dataset_name, self.tokenizer)

        dataset_name = os.path.join(args.data_name,"valid")
        data_path = "data/{}.json".format(dataset_name)
        self.validset, self.maps = loader.load_data_num(data_path, dataset_name, self.tokenizer)

        dataset_name = os.path.join(args.data_name,"test")
        data_path = "data/{}.json".format(dataset_name)
        self.testset, self.maps = loader.load_data_num(data_path, dataset_name, self.tokenizer)
        self.args = args
        self.batch = int(args.batch_size)
        self.epoch = int(args.epoch)
        self.seq_len = 512
        self.hid_dim = 256
        self.emb_dim = name2dim[args.model_name]

        self.threshold = []

        self.train_dataloader = DataLoader(dataset=self.trainset,
                                           batch_size=self.batch,
                                           shuffle=True,
                                           drop_last=False,)

        self.valid_dataloader = DataLoader(dataset=self.validset,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           drop_last=False,)
                                           
        self.test_dataloader = DataLoader(dataset=self.testset,
                                           batch_size=self.batch,
                                           shuffle=False,
                                           drop_last=False,)

        a_details = loader.load_details("code/utils/article_details.json", self.maps["a2i"], self.tokenizer, dataset_name, max_len = 200)
        c_details = loader.load_details("code/utils/charge_details.json", self.maps["c2i"], self.tokenizer, dataset_name, max_len = 200)

        details = {
            "a_details": a_details,
            "c_details": c_details,
            # "ar_adj":{
            #     "co": loader.load_article_adj(mode = "co"),
            #     "tfidf": loader.load_article_adj(mode = "tfidf"),
            # },
            # "ar_adj": loader.load_article_adj(mode = "co"),
            # "ch_adj": loader.load_charge_adj(),
        }
        # self.c2a = loader.load_c2a("data/filter/cail/meta/c2a_idx.json")

        for det in details.keys() & ["a_details", "c_details"]:
            for k in details[det]:
                details[det][k] = details[det][k].cuda()

        self.model = name2model[args.model_name](
            vocab_size=self.tokenizer.vocab_size, emb_dim=self.emb_dim, hid_dim=self.hid_dim, maps=self.maps, details = details)

        self.cur_time = time.strftime(
            '%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        self.model_name = "{}/{}".format(args.model_name, self.dataset_name)
        self.model_save_dir = "code/logs/{}/{}/".format(self.model_name, self.cur_time)
        print("model_save_dir: ", self.model_save_dir)

        self.task_name = ["charge", "article", "penalty"]
        # self.task_name = ["article", "penalty"]
        # self.task_name = ["charge", "article"]
        # self.task_name = ["penalty"]
        # self.task_name = ["article"]

        self.sub_task_name = ["c_attn", "a_attn", "c2a_constraint"]
        self.sub_task_name = ["dthreshold", "a_cos", "cl"]
        # self.sub_task_name = ["a_cos", "cl"]
        # self.sub_task_name = ["a_cos"]
        self.sub_task_name = ["cl"]
        self.sub_task_name = []

        if args.cl:
            self.sub_task_name.append("cl")

        self.article_task_name = ["penalty_ar", "charge_ar"]
        self.article_task_name = []

        print(self.model)
        print("train samples: ", len(self.trainset))
        print("valid samples: ", len(self.validset))

        
        params = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad]
        # non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')], 'lr': 1e-5}
        # bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': 1e-5}
        non_bert_params = {'params': [v for k, v in params if 'bert.' not in k], 'lr': 1e-4}
        bert_params = {'params': [v for k, v in params if 'bert.' in k], 'lr': 1e-5}
        self.optimizer = torch.optim.Adam([non_bert_params, bert_params])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # torch.functional.norm()

        self.loss_function = {
            # "article": nn.MultiLabelSoftMarginLoss(),
            # "charge": nn.MultiLabelSoftMarginLoss(),
            # "article": nn.BCEWithLogitsLoss(),
            # "charge": log_sum_loss,
            # "a_attn" : log_sum_loss,
            # "c_attn" : nn.CrossEntropyLoss(),
            # "c2a_constraint" : log_sum_loss,
            # "a2c_constraint" : log_sum_loss,

            # "penalty_ar": log_sum_loss,
            # "charge_ar": nn.CrossEntropyLoss(),

            "article": log_sum_loss,
            "charge": nn.CrossEntropyLoss(),
            "penalty": nn.CrossEntropyLoss(),
        }

        self.score_function = {
            "article": self.f1_score_macro,
            "charge": self.f1_score_macro,
            # "penalty": acc25,
            "penalty": self.f1_score_macro,
            "penalty_ar": self.f1_score_macro,
            "charge_ar": self.f1_score_macro,
        }

        # self.set_param_trainable(trainable=True)


        if args.load_path is not None:
            print("--- stage2 ---")
            print("load model path:", args.load_path)
            # self.model_name = self.model_name+"_s2"
            checkpoint = torch.load(args.load_path)
            
            model_load = checkpoint['model']
            load_model_dict = model_load.state_dict()
            load_model_dict = {k.replace("module.",""): v for k,v in load_model_dict.items()}
            cur_model_dict =  self.model.state_dict()
            state_dict = {k:v for k,v in load_model_dict.items() if k in cur_model_dict.keys()}
            noused_state_dict = {k:v for k,v in load_model_dict.items() if k not in cur_model_dict.keys()}
            noinit_state_dict = {k:v for k,v in cur_model_dict.items() if k not in load_model_dict.keys()}
            
            print("=== not used ===")
            print(noused_state_dict.keys())
            print("=== not init ===")
            print(noinit_state_dict.keys())

            cur_model_dict.update(state_dict)
            self.model.load_state_dict(cur_model_dict)

            # self.optimizer = checkpoint['optimizer']
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            self.evaluate(args.load_path, save_result=True)
            self.set_param_trainable(trainable=True)
            # self.model = self.model.module

        print("parameter counts: ", self.count_parameters())

        self.model = self.model.cuda()
        print("dp: {}  cl:{}".format(args.dp, args.cl))
        print("model_save_dir: ", self.model_save_dir)
        if args.dp:
            self.model = nn.DataParallel(self.model)

    def set_param_trainable(self, trainable):
        for name, param in self.model.named_parameters():
            # print(param.grad)
            param.requires_grad = trainable

    def check_param_grad(self):
        for name, parms in self.model.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def precision_macro(self, y_true, y_pred):
        ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return ma_p

    def recall_macro(self, y_true, y_pred):
        ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return ma_r

    def f1_score_macro(self, y_true, y_pred):
        mi_p, mi_r, mi_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        return (mi_p + mi_r + mi_f1 + ma_p + ma_r + ma_f1)/6

    def split_article(self, gt_ar):
        penalty_ar = gt_ar[:, :12]
        all0 = (penalty_ar.sum(-1) == 0).unsqueeze(-1).int()
        penalty_ar = torch.cat([penalty_ar, all0], dim = -1)
        charge_ar = gt_ar[:, 12:]
        _, charge_ar_idx = charge_ar.topk(1)

        return penalty_ar, charge_ar_idx.squeeze(-1)

    def train(self):
        best_score = -1
        writer = SummaryWriter(self.model_save_dir)
        early_stop_cnt = 0
        # writer.add_graph(model=self.model)
        for e in range(self.epoch):
            # train
            self.model.train()
            print("--- train ---")
            tq = tqdm(self.train_dataloader)
            tot_loss = 0
            cl_loss = 0
            for data in tq:
                # if data["article"].shape[0] == self.batch:
                #     continue
                for name in self.task_name:
                    data[name] = data[name].cuda()
                
                self.optimizer.zero_grad()
                out = self.model(data)

                loss = 0
                for name in self.task_name:
                    cur_loss = self.loss_function[name](out[name], data[name])
                    loss += cur_loss

                # for name in self.sub_task_name:
                #     if "cl" == name:
                #         continue
                #     label_name = "charge" if "c_" in name else "article"
                #     cur_loss = self.loss_function[name](out[name], data[label_name])
                #     loss += cur_loss
                
                for name in self.article_task_name:
                    cur_loss = self.loss_function[name](out[name], data[name])
                    loss += cur_loss

                tot_loss += loss.detach().cpu()
                if "cl" in self.sub_task_name:
                    af_emb = out["cl_emb"]["af_emb"]
                    a_emb = out["cl_emb"]["a_emb"]
                    cf_emb = out["cl_emb"]["cf_emb"]
                    c_emb = out["cl_emb"]["c_emb"]
                    gt_ar = data["article"].cuda()
                    gt_ch = data["charge"].cuda()

                    cl_loss = torch.tensor(0.).cuda()
                    cl_loss += get_contrastive_loss_sup(af_emb, gt_ar)
                    cl_loss += get_contrastive_loss_unsup(af_emb, a_emb, gt_ar)
                    cl_loss += get_contrastive_loss_sup(cf_emb, gt_ch)
                    cl_loss += get_contrastive_loss_unsup(cf_emb, c_emb, gt_ch)
                    cl_loss = cl_loss * 0.5
                    loss += cl_loss

                if "dthreshold" in self.sub_task_name:
                    dt_loss = dthreshold_loss(data["article"], out["article"], out["meta"]["threshold"])
                    loss += dt_loss

                if "a_cos" in self.sub_task_name:
                    a_loss = a_cos_loss(a_emb[0])
                    loss += a_loss
 
                argparams = {
                    "epoch": e,
                    "train_loss": np.around(loss.detach().cpu().numpy(), 4),
                }

                if "cl" in self.sub_task_name:
                    argparams["cl"] = np.around(cl_loss.detach().cpu().numpy(), 4)
                if "dthreshold" in self.sub_task_name:
                    argparams["dt"] = np.around(dt_loss.detach().cpu().numpy(), 4)
                if "a_cos" in self.sub_task_name:
                    argparams["a_cos"] = np.around(a_loss.detach().cpu().numpy(), 4)

                tq.set_postfix(**argparams)

                loss.backward()
                self.optimizer.step()
                # break

            writer.add_scalar("train loss", tot_loss, e)
            if "cl" in self.sub_task_name:
                writer.add_scalar("cl_loss", cl_loss, e)

            # valid
            savedStdout = sys.stdout 
            out_save_dir = "out/{}/".format(self.model_save_dir).replace("code/logs/","").replace("//","/")
            if not os.path.exists(out_save_dir):
                os.makedirs(out_save_dir)
            out_file = out_save_dir + 'out.txt'
            with open(out_file, 'a+') as file:
                sys.stdout = file 
                print(f"\n========== epoch: {e} ==========")
                print("--- valid ---")
                print("model_save_dir: ", self.model_save_dir)
                valid_out = self.infer(self.model, self.valid_dataloader)
                print("--- test ---")
                _ = self.infer(self.model, self.test_dataloader)
            sys.stdout = savedStdout 

            cur_score = 0
            for name in self.task_name + self.article_task_name:
                cur_task_score = self.score_function[name](valid_out[name]["true"], valid_out[name]["pred"])
                writer.add_scalar("valid {}".format(name), cur_task_score, e)
                cur_score += cur_task_score

            save_path = self.model_save_dir+"best_model.pt"
            if cur_score > best_score:
                best_score = cur_score
                if not os.path.exists(self.model_save_dir):
                    os.makedirs(self.model_save_dir)
                print("best model saved!")
                torch.save({"model": self.model, "optimizer": self.optimizer}, save_path)
                early_stop_cnt = 0
            # self.evaluate(save_path, save_result=False, evaluate_self_testset=True)
            early_stop_cnt += 1

            if early_stop_cnt > 20:
                break
            
    def infer_one(self, fact, eval_path):
        fact = " ".join(jieba.lcut(fact))
        fact = self.tokenizer([fact, fact], max_length=512, return_tensors="pt", padding="max_length", truncation=True)
        data = {
            "fact": fact,
        }
        checkpoint = torch.load(eval_path)
        model = checkpoint['model']
        print(model)
        res = model(data)
        print(res)


    def infer(self, model, data_loader, mode = "valid"):
        self.model.eval()
        infer_task_name = self.task_name
        tq = tqdm(data_loader)
        eval_out = {k: [] for k in infer_task_name}
        meta_out = {"af_emb":[], "a_emb":[], "cf_emb":[], "c_emb":[]} # af_emb shape:[batch, 70, dim]
        # model = model.module
        for data in tq:
            with torch.no_grad():
                out = model(data)
                # data["penalty_ar"], data["charge_ar"] = self.split_article(data["article"].cuda())

                for name in infer_task_name:
                    eval_out[name].append((out[name], data[name].cuda()))

                for name in out["meta"]:
                    if "threshold" == name:
                        continue
                    meta_out[name].append(out["meta"][name])
            # break

        for name in eval_out.keys():
            pred = torch.cat([i[0] for i in eval_out[name]])
            true = torch.cat([i[1] for i in eval_out[name]])
            eval_out[name] = {"pred": pred, "true": true}
        
        for name in meta_out:
            if len(meta_out[name]) == 0:
                break
            cur_merge = torch.cat(meta_out[name], dim = 0)
            meta_out[name] = cur_merge.cpu().numpy()
        eval_out["meta"] = meta_out

        for name in infer_task_name:
            print("=== {} ===".format(name))
            # if "single" in self.dataset_name:
            if name in ["charge", "penalty", "charge_ar"]: # single label
                pred = eval_out[name]["pred"].detach().cpu().numpy()
                y_pred = pred.argmax(-1)
            else: # multi label
                pred = eval_out[name]["pred"].detach().cpu()
                y_true = eval_out[name]["true"].detach().cpu().numpy()
                threshold = 0
                zero = torch.zeros_like(pred)
                one = torch.ones_like(pred)
                pred = torch.where(pred <= threshold, zero, pred)
                pred = torch.where(pred > threshold, one, pred)
                y_pred = pred.numpy()
                
            y_true = eval_out[name]["true"].detach().cpu().numpy()
            eval_out[name]["pred"] = y_pred
            eval_out[name]["true"] = y_true

            # if name =="article":
            #     y_pred = np.delete(y_pred, y_true.sum(0) == 0,  axis = 1)
            #     y_true = np.delete(y_true, y_true.sum(0) == 0,  axis = 1)

            def print_metrics(y_true, y_pred, average = "samples"):
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average = average)
                jaccard = metrics.jaccard_score(y_true, y_pred, average = average)
                print("{}: p: {}, r: {}, f1: {}, jac: {}".format(average,np.round(p* 100,6), np.round(r*100,6), np.round(f1*100,6), np.round(jaccard*100,6)))
            
            print_metrics(y_true, y_pred, "micro")
            print_metrics(y_true, y_pred, "macro")
            if name in ["article", "penalty_ar"]: # multi label
                print_metrics(y_true, y_pred, "samples")    
        
        return eval_out


    def evaluate(self, load_path, save_result = True,  evaluate_self_testset=False):
        """
            load_path: saved model path
            save_result: True or False. save result to csv file
        """
        print("--- evaluate on testset: ---")
        testset = self.testset

        print("test samples: ", len(testset))
        test_dataloader = DataLoader(dataset=testset,
                                     batch_size=self.batch,
                                     shuffle=False,
                                     drop_last=False)

        print("--- test ---")
        print("load model path: ", load_path)
        checkpoint = torch.load(load_path)
        model = checkpoint['model']
        # print(model)

        test_out = self.infer(model, test_dataloader)

        out_save_dir = "out/{}/".format(self.model_save_dir).replace("code/logs/","")

        if save_result:
            for name in self.task_name :
                if not os.path.exists(out_save_dir):
                    os.makedirs(out_save_dir)
                print("=== {} result saved ===".format(name))
                with open(os.path.join(out_save_dir, name + "_msg.json"),"w") as f:
                    msg = {
                        "pred": test_out[name]["pred"].tolist(),
                        "true": test_out[name]["true"].tolist()
                    }
                    if "true" in test_out[name]:
                        msg["true"] = test_out[name]["true"].tolist()
                    json.dump(msg, f)
            # meta = test_out["meta"]

            # for name in meta:
            #     fname = "out/{}/{}_msg.pt".format(self.model_name, name)
            #     torch.save(torch.tensor(meta[name]), fname)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu')
    parser.add_argument('--model_name', default='ELELJP_Num', help='model_name')
    parser.add_argument('--load_path', default=None, help='load model path')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--epoch', default=100, help='batch size')
    parser.add_argument('--cl', action='store_true', help='with contrastive learning')
    parser.add_argument('--dp', action='store_true', help='multi gpu')
    parser.add_argument('--data_name', default="filter/laic", help='')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainer = Trainer(args)
    trainer.train()

    print("== test_best_model ==")
    eval_path = trainer.model_save_dir + "best_model.pt"

    trainer.evaluate(
        eval_path,
        save_result=True,
        evaluate_self_testset=True
    )

