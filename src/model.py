import torch
from torch import nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.decoder import Decoder

class Model(nn.Module):

    def __init__(self, cfg, tokenizer):
        super(Model, self).__init__()
        self.cfg = cfg
        self.pad_id = tokenizer.pad_id
        self.unk_id = tokenizer.unk_id
        self.sos_id = tokenizer.sos_id
        self.eos_id = tokenizer.eos_id
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

    def __beam_search_first_step(self, enc_out, encoder_mask, beam_size):
        bs, mlen, d_model = enc_out.shape

        batch_trg = torch.full((bs, 1), self.sos_id, device=self.device, dtype=torch.long)
        decoder_mask = None

        decoder_out = self.decoder(batch_trg, enc_out, encoder_mask, decoder_mask)
        decoder_out = decoder_out[:, -1, :]
        logits = self.fc(decoder_out)

        blocking_list = torch.tensor([
            self.pad_id,
            self.unk_id,
            self.sos_id,
            self.eos_id
        ], device=self.device)
        logits[..., blocking_list] = float("-inf")

        log_probs = torch.log_softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(log_probs, beam_size, dim=-1)

        scores = top_scores.view((bs * beam_size, 1))
        sequences = top_indices.view((bs * beam_size, 1))
        sequences = torch.cat([
            torch.full((bs * beam_size, 1), self.sos_id, device=self.device, dtype=torch.long),
            sequences
        ], dim=-1)

        return scores, sequences

    def decoder_beam_search(self, enc_out, encoder_mask):
        print("hello beam search")
        beam_size = self.cfg.beam_size
        beam_max_len = self.cfg.beam_max_length
        bs, mlen, d_model = enc_out.shape

        scores, sequences = self.__beam_search_first_step(enc_out, encoder_mask, beam_size)
        breakpoint()

        for i in range(2, beam_max_len):
            self.decoder()


        breakpoint()



        for i in range(beam_max_len):
            decoder_out = self.decoder(batch_trg, enc_out, encoder_mask, decoder_mask)
            decoder_out = decoder_out[:, -1, :]
            log_probs = torch.log_softmax(self.fc(decoder_out), dim=-1)
            breakpoint()



        breakpoint()








    def decoder_beam_search_old(self, enc_out, encoder_mask):
        beam_size = self.cfg.beam_size
        beam_max_len = self.cfg.beam_max_length
        pad_id = self.pad_id
        sos_id = self.sos_id
        eos_id = self.eos_id

        bs, src_len, d_model = enc_out.shape
        vocab_size = self.fc.out_features

        # ---- Repeat encoder outputs for each beam ----
        enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
        enc_out = enc_out.view(bs * beam_size, src_len, d_model)

        if encoder_mask is not None:
            encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
            encoder_mask = encoder_mask.view(bs * beam_size, *encoder_mask.shape[2:])

        # ---- Initialize beams ----
        sequences = torch.full((bs * beam_size, 1), sos_id, dtype=torch.long, device=self.device)
        scores = torch.zeros(bs * beam_size, device=self.device)
        finished = torch.zeros(bs * beam_size, dtype=torch.bool, device=self.device)

        for t in range(beam_max_len):

            # Decoder forward
            decoder_out = self.decoder(sequences, enc_out, encoder_mask, None)
            step_out = decoder_out[:, -1, :]                            # [bs*beam, d_model]
            logits = self.fc(step_out)                                  # [bs*beam, vocab]
            log_probs = F.log_softmax(logits, dim=-1)                   # more stable

            # ---- Mask finished beams so they only generate PAD ----
            log_probs[finished] = -1e9
            log_probs[finished, pad_id] = 0

            # ---- Add previous beam scores ----
            total_scores = log_probs + scores.unsqueeze(1)              # [bs*beam, vocab]

            # ---- Reshape per batch ----
            total_scores = total_scores.view(bs, beam_size * vocab_size)

            # ---- Select top-k ----
            top_scores, top_indices = torch.topk(total_scores, beam_size, dim=-1)

            # Get beam + token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # ---- Reorder sequences & states ----
            offset = (torch.arange(bs, device=self.device) * beam_size).unsqueeze(1)
            flat_beam_idx = (beam_indices + offset).view(-1)

            sequences = sequences[flat_beam_idx]
            enc_out = enc_out[flat_beam_idx]
            if encoder_mask is not None:
                encoder_mask = encoder_mask[flat_beam_idx]

            # ---- Append new tokens ----
            sequences = torch.cat([sequences, token_indices.view(bs * beam_size, 1)], dim=-1)
            scores = top_scores.view(-1)

            # ---- Mark finished beams ----
            finished = finished[flat_beam_idx]
            finished |= (token_indices.view(-1) == eos_id)

            # ---- Early stop if all finished ----
            if finished.all():
                break

        # ---- Pick best beam per batch ----
        sequences = sequences.view(bs, beam_size, -1)
        scores = scores.view(bs, beam_size)

        best_idx = scores.argmax(dim=-1)  # [bs]
        best_sequences = sequences[torch.arange(bs), best_idx]

        return best_sequences












    def decoder_greedy_search(self, enc_out, encoder_mask):
        breakpoint()


    def forward(self, batch_src, trg_teacher=None):
        encoder_mask = self.create_encoder_mask(batch_src)
        enc_out = self.encoder(batch_src, encoder_mask)

        if trg_teacher is not None:
            out = self.decoder_teacher_forcing(trg_teacher, enc_out, encoder_mask)
        else:
            # raise Exception("erorr")
            out = self.decoder_beam_search(enc_out, encoder_mask)
        return out
