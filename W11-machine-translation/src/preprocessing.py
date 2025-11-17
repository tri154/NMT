import html
import re
import unicodedata
import os
import dill as pk
from tokenizer import Tokenizer

class Prepocessing:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg)
        if not os.path.exists(cfg.cache_file):
            res = self.__prepare_data()
            pk.dump(res, open(cfg.cache_file, "wb"), -1)
            self.cfg.logging("cached dataset.", is_printed=True)
        else:
            res = pk.load(open(cfg.cache_file, "rb"))
            self.cfg.logging("use cached dataset", is_printed=True)

        self.train_set = res['train']
        self.dev_set = res['dev']
        self.test_set = res['test']

    def __clean_text(self, s):
        s = str(s)
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
        s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def __load_data(self, src, trg):
        src_sents = open(src, "r").readlines()
        trg_sents = open(trg, "r").readlines()
        assert len(src_sents) == len(trg_sents)
        res_src, res_trg = list(), list()
        for s, t in zip(src_sents, trg_sents):
            s = self.__clean_text(s)
            t = self.__clean_text(t)
            res_src.append(s)
            res_trg.append(t)
        return {"source": res_src,
                "target": res_trg}

    def __build_vocab(self, train_set):
        src = train_set["source"]
        trg = train_set["target"]
        src_tkn, src_vocab = self.tokenizer.tokenize_and_build_vocab(src)
        trg_tkn, trg_vocab = self.tokenizer.tokenize_and_build_vocab(trg)
        return src_tkn, src_vocab, trg_tkn, trg_vocab

    def __prepare_data(self):
        res = dict()
        # load data
        for key, value in self.cfg.files.items():
            src, target = value
            res[key] = self.__load_data(src, target)
        src_tkn, src_vocab, trg_tkn, trg_vocab = self.__build_vocab(res['train'])
        res['train'] = {
           "source": src_tkn,
            "target": trg_tkn }
        for key, value in self.cfg.files.items():
            src, target = value["source"], value["target"]
            src_tkn = self.tokenizer.tokenize_from_vocab(src, src_vocab)
            trg_tkn = self.tokenizer.tokenize_from_vocab(src, src_vocab)
        return res
