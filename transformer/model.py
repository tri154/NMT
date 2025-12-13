import torch
from torch import nn
from transformer import Encoder
from transformer import Decoder

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


    def __beam_search_first_step(self, enc_out, encoder_mask, beam_size, beam_max_len):
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
        sequences = top_indices.view((bs * beam_size))
        full_sequences = torch.full((bs * beam_size, beam_max_len), self.pad_id, device=self.device, dtype=torch.long)
        full_sequences[:, 0] = self.sos_id
        full_sequences[:, 1] = sequences

        return scores, full_sequences


    def decoder_beam_search(self, enc_out, encoder_mask):
        beam_size = self.cfg.beam_size
        bs, enc_mlen, d_model = enc_out.shape
        beam_max_len = enc_mlen + self.cfg.beam_max_length

        scores, full_sequences = self.__beam_search_first_step(enc_out, encoder_mask, beam_size, beam_max_len)
        seqs_len= torch.ones(bs * beam_size, device=self.device).unsqueeze(1)
        finished = torch.zeros(bs * beam_size, dtype=bool, device=self.device)

        enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
        enc_out = enc_out.view(bs * beam_size, *enc_out.shape[2:])
        encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
        encoder_mask = encoder_mask.view(bs * beam_size, *encoder_mask.shape[2:])

        blocking_list = torch.tensor([self.pad_id, self.unk_id, self.sos_id], device=self.device, dtype=torch.long)
        for len in range(2, beam_max_len):
            sequences = full_sequences[:, :len]
            decoder_mask = self.create_decoder_mask(sequences)
            decoder_out = self.decoder(sequences, enc_out, encoder_mask, decoder_mask)

            decoder_out = decoder_out[:, -1]
            logits = self.fc(decoder_out)
            logits[..., blocking_list] = float("-inf")

            log_probs = torch.log_softmax(logits, dim=-1)

            if finished.any():
                log_probs[finished] = - 1e-9
                log_probs[finished, self.pad_id] = 0

            seqs_len[~finished] += 1
            vocab_size = log_probs.shape[-1]

            total_scores = (log_probs + scores) / seqs_len # normalized in case finished.
            total_scores = total_scores.view(bs, beam_size * vocab_size)
            top_scores, top_indices = torch.topk(total_scores, beam_size, dim=-1)

            # update selected sequences
            selected_seqs = top_indices // vocab_size
            offsets = torch.arange(bs, device=self.device).unsqueeze(1) * beam_size
            selected_seqs = (selected_seqs + offsets).view(-1)
            full_sequences = full_sequences[selected_seqs]

            # update seqs_len (should be updated before scores)
            seqs_len = seqs_len[selected_seqs]

            # update scores (unormalized)
            scores = top_scores.view(-1, 1) * seqs_len

            # update selected tokens
            selected_tokens = top_indices % vocab_size
            selected_tokens = selected_tokens.view(-1)

            # update new sequences
            full_sequences[:, len] = selected_tokens

            # update finished
            finished = finished[selected_seqs]
            finished = finished | (selected_tokens == self.eos_id)

            if finished.all():
                break

        full_sequences = full_sequences.view(bs, beam_size, -1)
        scores = scores / seqs_len
        scores = scores.view(bs, beam_size)

        best_idx = scores.argmax(dim=-1)
        best_sequences = full_sequences[torch.arange(bs, device=self.device), best_idx]

        return best_sequences


    def forward(self, batch_src, trg_teacher=None):
        encoder_mask = self.create_encoder_mask(batch_src)
        enc_out = self.encoder(batch_src, encoder_mask)

        if trg_teacher is not None:
            out = self.decoder_teacher_forcing(trg_teacher, enc_out, encoder_mask)
        else:
            out = self.decoder_beam_search(enc_out, encoder_mask)
        return out
