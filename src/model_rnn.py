import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder

class Model(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Model, self).__init__()
        self.cfg = cfg
        self.pad_id = tokenizer.pad_id
        self.sos_id = tokenizer.sos_id
        self.eos_id = tokenizer.eos_id
        self.trg_vocab = tokenizer.trg_vocab

        self.encoder = Encoder(cfg, tokenizer)
        self.decoder = Decoder(cfg, tokenizer)
        self.fc = nn.Linear(cfg.emb_dim, len(self.trg_vocab), bias=False)

        # weight tying
        self.fc.weight = self.decoder.embedding.weight


    def decoder_teacher_forcing(self, batch_trg, enc_out, enc_hidden, src_mask):
        bs, trg_mlen = batch_trg.shape
        decoder_embs = list()

        for i in range(0, trg_mlen):
            if i == 0:
                hidden = enc_hidden[-1].unsqueeze(0) # ensure last layer
                decoder_input = batch_trg[:, 0].unsqueeze(-1)

            out, hidden = self.decoder(decoder_input, hidden, enc_out, src_mask)
            decoder_embs.append(out)

            if i != trg_mlen - 1:
                # teacher forcing
                decoder_input = batch_trg[:, i].unsqueeze(-1)

        decoder_embs = torch.cat(decoder_embs, dim=1)
        logits = self.fc(decoder_embs)
        return logits

    def decoder_greedy(self, enc_out, enc_hidden, src_lengths):
        device = self.cfg.device
        bs, _, _ = enc_out.shape

        batch_res = list()
        for sid in range(bs):
            s_res = list()
            s_out_length = src_lengths[sid]
            s_out = enc_out[sid:sid+1, :s_out_length]
            s_hidden = enc_hidden[:, sid:sid+1, :] # keep dim
            for tid in range(0, self.cfg.max_len_trg):
                if tid == 0:
                    de_input = torch.tensor([[self.sos_id]]).to(device)
                    s_res.append(self.sos_id)
                    de_hidden = s_hidden
                    s_out = s_out
                de_out, de_hidden = self.decoder(de_input, de_hidden, s_out, src_mask=None)
                logits = self.fc(de_out)
                # update
                de_input = torch.argmax(logits, dim=-1)
                s_res.append(de_input.item())
                if tid == self.cfg.max_len_trg - 1:
                    s_res.append(self.eos_id) # append eos
                    s_res = torch.tensor(s_res)
                if de_input.item() == self.eos_id:
                    s_res = torch.tensor(s_res)
                    break

            batch_res.append(s_res)
        return batch_res

    def forward(self, batch_src, batch_trg=None, is_training=True):
        # for training only.
        bs, src_max_len = batch_src.shape
        if is_training: bs, trg_max_len = batch_trg.shape

        src_lengths = (batch_src != self.pad_id).to(int).sum(dim=1)
        if is_training: trg_lengths = (batch_trg != self.pad_id).to(int).sum(dim=1)

        src_mask = torch.arange(src_max_len).expand((bs, src_max_len)) >= src_lengths.unsqueeze(1)
        src_mask = torch.masked_fill(src_mask.float(), src_mask, float("-inf"))

        enc_out, enc_hidden = self.encoder(batch_src, src_lengths)

        if is_training:
            logits = self.decoder_teacher_forcing(batch_trg, enc_out, enc_hidden, src_mask)
            return logits, trg_lengths
        else:
            # expected output: like [bs, mlen] token ids of translated sentences.
            out = self.decoder_greedy(enc_out, enc_hidden, src_lengths)
            return out
