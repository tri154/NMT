from collections import Counter
class Tokenizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.additional_tokens = [self.pad, self.unk, self.sos, self.eos]

    def tokenize(self, data):
        if isinstance(data, list):
            return [s.strip().split() for s in data]
        if isinstance(data, str):
            return data.strip().split()
        raise Exception("Invalid type.")

    def tokenize_from_vocab(self, data, vocab):
        # CONTINUE
        self.tokenize(data)

    def tokenize_and_build_vocab(self, data):
        tokenized = self.tokenize(data)
        counter = Counter()
        for s in tokenized:
            counter.update(s)
        if self.cfg.min_freq != -1:
            vocab = {word for word, freq in counter.items() if freq >= self.cfg.min_freq}
        else:
            vocab = counter.keys()
        vocab = self.additional_tokens + list(vocab)
        return tokenized, vocab
