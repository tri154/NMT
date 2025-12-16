from collections import Counter
from tqdm import tqdm
from shared import Tokenizer
import dill as pk

class WordTokenizer(Tokenizer):
    # Deprecated, now use shared vocab.

    def __init__(self, cfg):
        super().__init__(cfg)
        raise Exception("Deprecated")
        self.prepare_special_tokens()
        self.src_model = cfg.src_tkn_path + ".pkl"
        self.trg_model = cfg.trg_tkn_path + ".pkl"

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

    def tokenize_with_vocab(self, data, tag=None):
        res = list()
        for s in tqdm(data):
            temp = self.tokenize(s)
            temp = [self.sos] + temp + [self.eos]
            res.append(temp)
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

        vocab = self.additional_tokens + sorted(list(vocab))

        if tag == "source":
            self.src_vocab = vocab
        elif tag == "target":
            self.trg_vocab = vocab
        else:
            raise Exception("Invalid tag.")

        token2id, id2token = self.build_addition(tag)
        addition = {"vocab": vocab,
                "token2id": token2id,
                "id2token": id2token}
        path = self.src_model if tag == "source" else self.trg_model
        pk.dump(addition, open(path, "wb"), -1)
        self.cfg.logging(f"save to {path}")

    def token2ids(self, data, tag):
        assert tag in ["source", "target"]
        token2id = self.src_token2id if tag == "source" else self.trg_token2id

        return [
            [token2id.get(tok, self.unk_id) for tok in sent]
            for sent in data
        ]


    def detokenize(self, data, tag="target"):
        if tag == "target":
            id2token = self.trg_id2token
        else:
            id2token = self.src_id2token

        sentences = []

        for sent_ids in data:
            tokens = []

            for tid in sent_ids:
                if tid == self.eos_id:
                    break
                if tid == self.sos_id:
                    continue

                tok = id2token.get(int(tid), self.unk)
                tokens.append(tok)

            sentences.append(" ".join(tokens))
            # sentences.append(tokens)

        return sentences

    def load(self):
        src_addition = pk.load(open(self.src_model, "rb"))
        trg_addition = pk.load(open(self.trg_model, "rb"))
        self.assign_addition(src_addition, trg_addition)


    # self define

    def assign_addition(self, src_addition, trg_addition):
        """
        src_addition, trg_addition: dict
        keys -> value:
            vocab -> np.array
            token2id -> dict: str -> int
            id2token --> dict: int -> str
        return None
        """
        for key, value in src_addition.items():
            setattr(self, "src_" + key, value)
        for key, value in trg_addition.items():
            setattr(self, "trg_" + key, value)



    def build_addition(self, tag):
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
