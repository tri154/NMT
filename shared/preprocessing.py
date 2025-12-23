import html
import re
import unicodedata
import os
import dill as pk

class Prepocessing:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
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

        if os.path.exists(self.tokenizer.model_path):
            self.tokenizer.load()
        else:
            raise Exception("data loaded, not found tokenizer save.")


        # logging
        log = ""
        for key, value in res.items():
            if key in ["train", "dev", "test"]:
                sz = len(value['source'])
                log += f"Size of {key}: {sz} .\n"
            else:
                sz = len(value["vocab"])
                log += f"Size of {key}.vocab: {sz} .\n"
        self.cfg.logging(log, is_printed=True)


    def clean_text_simple(self, s, lowercase=False):
        s = str(s)
        s = html.unescape(s)

        s = re.sub(r"<[^>]+>", " ", s)

        s = unicodedata.normalize("NFKC", s)

        if lowercase:
            s = s.lower()

        s = re.sub(r"\s+", " ", s).strip()

        return s


    def __clean_text(self, s):
        # return self.clean_text_simple(s, lowercase=self.cfg.lowercase)
        lowercase = self.cfg.lowercase
        s = str(s)
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
        # s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
        if lowercase:
            s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def __load_data(self, src, trg, filter=False):
        max_len = self.cfg.train_max_len
        num_examples = self.cfg.num_examples

        src_sents = open(src, "r").readlines()
        trg_sents = open(trg, "r").readlines()
        assert len(src_sents) == len(trg_sents)
        res_src, res_trg = list(), list()

        for s, t in zip(src_sents, trg_sents):
            s = self.__clean_text(s)
            t = self.__clean_text(t)
            not_drop = max_len < 0
            not_drop = not_drop or (len(s.split()) <= max_len and len(t.split()) <= max_len)
            if not_drop or not filter:
                res_src.append(s)
                res_trg.append(t)
        if num_examples > 0:
            res_src = res_src[:num_examples]
            res_trg = res_trg[:num_examples]
        return {"source": res_src,
                "target": res_trg}

    def __build_vocab(self, train_set):
        src = train_set["source"]
        trg = train_set["target"]
        self.tokenizer.build_and_save_vocab(src, trg)

    def __prepare_data(self):
        res = dict()
        for key, value in self.cfg.files.items():
            src, target = value
            filter = key == "train"
            res[key] = self.__load_data(src, target, filter)

        self.__build_vocab(res['train'])
        self.tokenizer.load()

        for key, value in res.items():
            src, trg = value["source"], value["target"]
            src_tkn = self.tokenizer.tokenize_with_vocab(src)
            if key == "train":
                trg_tkn = self.tokenizer.tokenize_with_vocab(trg)
            else:
                # trg_tkn = self.tokenizer.tokenize(trg)
                trg_tkn = trg
            res[key] = {"source": src_tkn,
                        "target": trg_tkn}

        return res
