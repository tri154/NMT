from abc import ABC, abstractmethod

class Tokenizer(ABC):
    # Deprecated, now use shared vocab.

    def __init__(self, cfg):
        raise Exception("Deprecated")
        self.cfg = cfg
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.sos = "<sos>"
        self.eos = "<eos>"

        self.pad_id = None
        self.unk_id = None
        self.sos_id = None
        self.eos_id = None

        self.src_vocab = None
        self.src_token2id = None
        self.src_id2token = None

        self.trg_vocab = None
        self.trg_token2id = None
        self.trg_id2token = None


    @abstractmethod
    def prepare_special_tokens(self):
        self.pad_id = None
        self.unk_id = None
        self.sos_id = None
        self.eos_id = None

        self.additional_tokens = [self.pad, self.unk, self.sos, self.eos]
        self.additional_ids = [self.pad_id, self.unk_id, self.sos_id, self.eos_id]


    @abstractmethod
    def tokenize(self, line):
        """
        line: str or list[str]
        vocab: np.array
        return list or list[list]
        """
        pass


    @abstractmethod
    def tokenize_with_vocab(self, data, tag):
        """
        data: list[str]
        tag: str, 'source' or 'target', vocab to use.

        use tokenize method, filter post-processing when vocab is known.

        return list[list[str]]
        """
        pass


    @abstractmethod
    def build_and_save_vocab(self, data, tag):
        """
        data: list[str]
        tag: str, 'source' or 'target' save place.

        build vocab, use build_addition to build additional data.

        return {"vocab": vocab,
                "token2id": token2id,
                "id2token": id2token}
        """
        pass


    @abstractmethod
    def detokenize(self, data):
        """
        data: list[list[int]]
        return list[str]
        """
        pass

    @abstractmethod
    def token2ids(self, data, tag):
        """
        data: list[list[str]]
        tag: str 'source' or 'target', vocab to use
        return list[list[int]]
        """

    @abstractmethod
    def load(self):
        pass
