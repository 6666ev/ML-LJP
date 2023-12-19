from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import torch
import pickle
import json

class MyTokenizer:
    def __init__(self, embedding_path="code/gensim_train/word2vec.model") -> None:
        model = Word2Vec.load(embedding_path)
        self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.id2word = self.special_tokens + model.wv.index_to_key

        self.word2id = model.wv.key_to_index  # (8507)
        for k in self.word2id.keys():
            self.word2id[k] += len(self.special_tokens)
        for i in range(len(self.special_tokens)):
            self.word2id[self.special_tokens[i]] = i
        self.embedding_path = embedding_path
        self.vector_size = model.wv.vector_size
        self.vocab_size = len(self.word2id)
        special_token_vec = np.zeros(
            (len(self.special_tokens), self.vector_size))
        self.vectors = model.wv.vectors  # (8507, 300)
        self.vectors = np.concatenate(
            (special_token_vec, self.vectors))  # (8511, 300)

    def load_embedding(self):
        return self.vectors

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, sents, max_length=512, return_tensors="ls", padding="max_length", truncation=True):
        # padding和truncation这两个参数只是为了凑数，跟transformers的AutoTokenizer接口统一
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for sent in sents:
            sent = sent.split(" ")
            sent = [self.word2id[w] if w in self.word2id.keys() else self.word2id["[UNK]"] for w in sent]
            sent = [self.word2id["[SOS]"]] + sent + [self.word2id["[EOS]"]]
            sent_len = len(sent)
            sent += [0] * max_length
            sent = sent[:max_length]

            input_ids.append(sent)
            token_type_ids.append([0] * max_length)
            mask = [1] * sent_len + [0] * max_length
            attention_mask.append(mask[:max_length])

        if return_tensors == "np":
            input_ids = np.array(input_ids)
            token_type_ids = np.array(token_type_ids)
            attention_mask = np.array(attention_mask)
        elif return_tensors == 'pt':
            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
            attention_mask = torch.LongTensor(attention_mask)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token):
        res = []
        sent = []
        for t in token:
            if t == 0:
                break
            if t > len(self.id2word):
                sent.append("[UNK]")
            else:
                sent.append(self.id2word[t])
        sent = " ".join(sent)
        return res



class THULAC_Tokenizer:
    def __init__(self) -> None:
        embedding_path = "code/thulac_w2vec/cail_thulac.npy"
        w2i_path = "code/thulac_w2vec/w2id_thulac.pkl"
        self.word2id = pickle.load(open(w2i_path, 'rb'))

        word_embedding = np.cast[np.float32](np.load(embedding_path))
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embedding_path = embedding_path
        self.vocab_size, self.vector_size = word_embedding.shape
        self.vectors = word_embedding  # (16w, 200)

    def load_embedding(self):
        return self.vectors

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, sents, max_length=512, return_tensors="ls", padding="max_length", truncation=True):
        # padding和truncation这两个参数只是为了凑数，跟transformers的AutoTokenizer接口统一
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for sent in sents:
            sent = sent.split(" ")
            sent = [self.word2id[w] if w in self.word2id.keys() else self.word2id["UNK"] for w in sent]
            sent =  sent 
            sent_len = len(sent)
            sent += [self.word2id["BLANK"]] * max_length
            sent = sent[:max_length]

            input_ids.append(sent)
            token_type_ids.append([0] * max_length)
            mask = [1] * sent_len + [0] * max_length
            attention_mask.append(mask[:max_length])

        if return_tensors == "np":
            input_ids = np.array(input_ids)
            token_type_ids = np.array(token_type_ids)
            attention_mask = np.array(attention_mask)
        elif return_tensors == 'pt':
            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
            attention_mask = torch.LongTensor(attention_mask)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }

    def decode(self, token):
        sent = []
        for t in token:
            if t == self.word2id["BLANK"]:
                break
            if t > len(self.id2word):
                sent.append("UNK")
            else:
                sent.append(self.id2word[t])
        sent = " ".join(sent)
        return sent


if __name__ == "__main__":
    tokenizer = THULAC_Tokenizer()
    tokens = tokenizer.encode(["被告人 吴 XX 在 浙江 通过 朋友 认识 了 被害人 马 某丙 ， 后 隐瞒 已婚 事实 ， 谎称 自己 是 开 物流 公司 的 ， 以 恋爱 为名 追求 马 某丙 。"])
    sents = tokenizer.decode(tokens["input_ids"][0])

    print(tokens["input_ids"])
    print(sents)
