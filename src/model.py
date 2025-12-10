import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder

class Model(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Model, self).__init__()
        self.cfg = cfg
        self.pad_id = tokenizer.pad_id
        self.device = cfg.device

        self.encoder = Encoder(cfg, tokenizer)
        self.decoder = Decoder(cfg, tokenizer)

        self.fc = nn.Linear(cfg.d_model, len(tokenizer.trg_vocab), bias=False)
        self.fc.weight = self.decoder.embedding.weight

    def create_encoder_mask(self, batch_src):
        bs, mlen = batch_src.shape
        src_lengths = (batch_src != self.pad_id).to(int).sum(dim=1)
        padding_mask = torch.arange(mlen, device=self.device).expand(bs, mlen) >= src_lengths.unsqueeze(-1)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        return padding_mask


    def create_decoder_mask(self, batch_trg):
        bs, mlen = batch_trg.shape
        mask = torch.triu(torch.ones(mlen, mlen, device=self.device), diagonal=1)
        return mask.bool().unsqueeze(0).unsqueeze(1)


    def decoder_teacher_forcing(self, batch_trg, enc_out, encoder_mask):
        decoder_mask = self.create_decoder_mask(batch_trg)
        decoder_out = self.decoder(batch_trg, enc_out, encoder_mask, decoder_mask)

        logits = self.fc(decoder_out)
        return logits


    def decoder_beam_search(self, enc_out, encoder_mask):
        breakpoint()


    def forward(self, batch_src, trg_teacher=None):
        encoder_mask = self.create_encoder_mask(batch_src)
        enc_out = self.encoder(batch_src, encoder_mask)

        if trg_teacher is not None:
            out = self.decoder_teacher_forcing(trg_teacher, enc_out, encoder_mask)
        else:
            raise Exception("error")
        return out
