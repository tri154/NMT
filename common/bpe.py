import os
import sentencepiece as spm
import numpy as np
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.sos = "<sos>"
        self.eos = "<eos>"

        self.sp = spm.SentencePieceProcessor()

        self.prefix_name = self.cfg.tkn_prefix
        self.model_prefix = os.path.join(self.cfg.result_path, self.prefix_name)
        self.model_path = self.model_prefix + ".model"

        self.prepare_special_tokens()

    def prepare_special_tokens(self):
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

        self.additional_tokens = [self.pad, self.unk, self.sos, self.eos]
        self.additional_ids = [self.pad_id, self.unk_id, self.sos_id, self.eos_id]

    def tokenize(self, line):
        if isinstance(line, str):
            return self.sp.encode_as_pieces(line)
        elif isinstance(line, list):
            return [self.sp.encode_as_pieces(s) for s in line]
        raise ValueError("Invalid type for tokenize.")

    def tokenize_with_vocab(self, data):
        sp_model = self.sp

        res = []
        for s in tqdm(data, desc="Tokenizing"):
            pieces = sp_model.encode_as_pieces(s)
            pieces = [self.sos] + pieces + [self.eos]
            res.append(pieces)

        return res

    def build_and_save_vocab(self, src_data, trg_data):
        temp_file = f"{self.prefix_name}_train_data.txt"
        with open(temp_file, "a", encoding="utf-8") as f:
            for line in src_data:
                f.write(line + "\n")
        with open(temp_file, "a", encoding="utf-8") as f:
            for line in trg_data:
                f.write(line + "\n")

        vocab_size = self.cfg.vocab_size

        cmd = (
            f"--input={temp_file} "
            f"--model_prefix={self.model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type=bpe "
            f"--character_coverage=1.0 "
            f"--pad_id={self.pad_id} "
            f"--unk_id={self.unk_id} "
            f"--bos_id={self.sos_id} "
            f"--eos_id={self.eos_id} "
            f"--pad_piece={self.pad} "
            f"--unk_piece={self.unk} "
            f"--bos_piece={self.sos} "
            f"--eos_piece={self.eos}"
        )

        # if hasattr(self.cfg, "spm_extra_args"):
        #     cmd += f" {self.cfg.spm_extra_args}"

        spm.SentencePieceTrainer.train(cmd)
        if os.path.exists(temp_file):
            os.remove(temp_file)


    def token2ids(self, data):
        sp_model = self.sp

        def tokens_to_ids(tokens):
            return [sp_model.piece_to_id(t) for t in tokens]

        return [tokens_to_ids(sent) for sent in data]

    def detokenize(self, data):
        decoded_sentences = []
        for sent_ids in data:
            if isinstance(sent_ids, np.ndarray):
                sent_ids = sent_ids.tolist()

            decoded = self.sp.decode_ids(sent_ids)
            decoded_sentences.append(decoded)

        return decoded_sentences

    def load(self):
        self.sp.load(self.model_path)
        sp_model = self.sp

        vocab_list = []

        for i in range(sp_model.get_piece_size()):
            piece = sp_model.id_to_piece(i)
            vocab_list.append(piece)

        vocab_array = np.array(vocab_list)

        self.vocab = vocab_array
