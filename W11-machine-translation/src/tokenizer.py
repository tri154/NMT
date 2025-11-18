from collections import Counter
import numpy as np
from tqdm import tqdm
class Tokenizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.additional_tokens = [self.pad, self.unk, self.sos, self.eos]

        # assign later.
        self.src_vocab = None
        self.src_token2id = None
        self.src_id2token = None

        self.trg_vocab = None
        self.trg_token2id = None
        self.trg_id2token = None


    def assign(self, src_addition, trg_addition):
        for key, value in src_addition.items():
            setattr(self, "src_" + key, value)
        for key, value in trg_addition.items():
            setattr(self, "trg_" + key, value)

    def tokenize(self, data):
        if isinstance(data, list):
            return [s.strip().split() for s in data]
        if isinstance(data, str):
            return data.strip().split()
        raise Exception("Invalid type.")

    def tokenize_with_vocab(self, data, tag):
        if tag == "source":
            vocab = self.src_vocab
        elif tag == "target":
            vocab = self.trg_vocab
        else:
            raise Exception("Invalid tag.")

        res = list()
        for s in tqdm(data):
            temp = self.tokenize(s)
            temp = [self.sos] + temp + [self.eos]
            temp = np.array(temp)
            mask = np.isin(temp, vocab)
            temp[~mask] = self.unk
            res.append(temp.tolist())
        return res

    def build_and_save_vocab(self, data, tag):
        tokenized = self.tokenize(data)
        counter = Counter()
        for s in tokenized:
            counter.update(s)
        if self.cfg.min_freq != -1:
            vocab = {word for word, freq in counter.items() if freq >= self.cfg.min_freq}
        else:
            vocab = counter.keys()
        vocab = self.additional_tokens + list(vocab)
        vocab = np.array(vocab)

        if tag == "source":
            self.src_vocab = vocab
        elif tag == "target":
            self.trg_vocab = vocab
        else:
            raise Exception("Invalid tag.")

        token2id, id2token = self.__build_addition(tag)
        return {"vocab": vocab,
                "token2id": token2id,
                "id2token": id2token}

    def __build_addition(self, tag):
        if tag == "source":
            if self.src_token2id is None:
                self.src_token2id = {t: idx for idx, t in enumerate(self.src_vocab)}
                self.src_id2token = {idx: t for idx, t in enumerate(self.src_vocab)}
                return self.src_token2id, self.src_id2token
        elif tag == "target":
            if self.trg_token2id is None:
                self.trg_token2id = {t: idx for idx, t in enumerate(self.trg_vocab)}
                self.trg_id2token = {idx: t for idx, t in enumerate(self.trg_vocab)}
                return self.trg_token2id, self.trg_id2token
        else:
            raise Exception("Invalid tag.")

    def token2ids(self, data, tag):
        data = np.array(data)
        if tag == "source":
            fn = np.vectorize(self.src_token2id.get)
        elif tag == "target":
            fn = np.vectorize(self.trg_token2id.get)
        else:
            raise Exception("Invalid tag.")

        data = fn(data)
        return data.tolist()
