import os
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
from common import Tokenizer

class BPETokenizer(Tokenizer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sp_src = spm.SentencePieceProcessor()
        self.sp_trg = spm.SentencePieceProcessor()

        # Paths to save the trained models
        self.src_model_prefix = cfg.src_model_prefix
        self.trg_model_prefix = cfg.trg_model_prefix

        self.prepare_special_tokens()

    def prepare_special_tokens(self):
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3

        self.additional_tokens = [self.pad, self.unk, self.sos, self.eos]
        self.additional_ids = [self.pad_id, self.unk_id, self.sos_id, self.eos_id]

    def _get_model(self, tag):
        if tag == "source":
            return self.sp_src
        elif tag == "target":
            return self.sp_trg
        else:
            raise ValueError("Invalid tag. Must be 'source' or 'target'.")

    def tokenize(self, line):
        if isinstance(line, str):
            return self.sp_src.encode_as_pieces(line)
        elif isinstance(line, list):
            return [self.sp_src.encode_as_pieces(s) for s in line]
        raise ValueError("Invalid type for tokenize.")

    def tokenize_with_vocab(self, data, tag):
        sp_model = self._get_model(tag)

        res = []
        for s in tqdm(data, desc=f"Tokenizing {tag}"):
            pieces = sp_model.encode_as_pieces(s)
            pieces = [self.sos] + pieces + [self.eos]
            res.append(pieces)

        return res

    def build_and_save_vocab(self, data, tag):
        prefix = self.src_model_prefix if tag == "source" else self.trg_model_prefix

        temp_file = f"{prefix}_train_data.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")

        vocab_size = getattr(self.cfg, f"{tag}_vocab_size")

        cmd = (
            f"--input={temp_file} "
            f"--model_prefix={prefix} "
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

        sp_model = self._get_model(tag)
        sp_model.load(f"{prefix}.model")

        vocab_list = []
        token2id = {}
        id2token = {}

        for i in range(sp_model.get_piece_size()):
            piece = sp_model.id_to_piece(i)
            vocab_list.append(piece)
            token2id[piece] = i
            id2token[i] = piece

        vocab_array = np.array(vocab_list)

        if tag == "source":
            self.src_vocab = vocab_array
        else:
            self.trg_vocab = vocab_array

        if os.path.exists(temp_file):
            os.remove(temp_file)

        self.assign_addition(
            {tag: {"token2id": token2id, "id2token": id2token}} if tag=="source" else {},
            {tag: {"token2id": token2id, "id2token": id2token}} if tag=="target" else {}
        )
        return {
            "vocab": vocab_array,
            "token2id": token2id,
            "id2token": id2token
        }

    def token2ids(self, data, tag):
        sp_model = self._get_model(tag)

        def tokens_to_ids(tokens):
            return [sp_model.piece_to_id(t) for t in tokens]

        return [tokens_to_ids(sent) for sent in data]

    def detokenize(self, data):
        decoded_sentences = []
        for sent_ids in data:
            if isinstance(sent_ids, np.ndarray):
                sent_ids = sent_ids.tolist()

            decoded = self.sp_trg.decode_ids(sent_ids)
            decoded_sentences.append(decoded)

        return decoded_sentences
