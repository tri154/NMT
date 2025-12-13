from collections import Counter
import numpy as np
from tqdm import tqdm
from common import Tokenizer

class WordTokenizer(Tokenizer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.prepare_special_tokens()

    def prepare_special_tokens(self):
        # additional tokens share the same id in src and trg.
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

        self.additional_tokens = [self.pad, self.unk, self.sos, self.eos]
        self.additional_ids = [self.pad_id, self.unk_id, self.sos_id, self.eos_id]

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

        token2id, id2token = self.build_addition(tag)
        return {"vocab": vocab,
                "token2id": token2id,
                "id2token": id2token}

    def token2ids(self, data, tag):
        assert tag in ["source", "target"]
        token2id_get = self.src_token2id.get if tag == "source" else self.trg_token2id.get
        fn = np.vectorize(token2id_get)

        data = fn(np.array(data))
        data = data.tolist()
        return data

    def detokenize(self, data):
        fn = np.vectorize(self.trg_id2token.get)
        tokens = fn(data)
        keep_mask = ~np.isin(data, np.array(self.additional_ids))


        return [
            " ".join(sent_token[mask])
            for sent_token, mask in zip(tokens, keep_mask)
        ]

        # res = []
        # for sent in data:
        #     sent = sent.cpu().numpy()
        #     tokens = fn(sent).tolist()
        #     tokens = [tok for tok in tokens if tok not in self.additional_tokens]
        #     res.append(" ".join(tokens))
        # return res
