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

        self.fc = nn.Linear(cfg.d_model, len(tokenizer.trg_vocab))
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def create_encoder_mask(self, batch_src):
        bs, mlen = batch_src.shape
        padding_mask = (batch_src == self.pad_id).unsqueeze(1).unsqueeze(2)
        return padding_mask


    def create_decoder_mask(self, batch_trg):
        bs, mlen = batch_trg.shape
        mask = torch.triu(torch.ones(mlen, mlen, device=self.device), diagonal=1)
        mask = mask.bool().unsqueeze(0).unsqueeze(1)
        # mask = mask.bool()
        # padding_mask = (batch_trg == self.pad_id).unsqueeze(1).unsqueeze(2)
        # mask = mask | padding_mask
        return mask


    def decoder_teacher_forcing(self, batch_trg, enc_out, encoder_mask):
        decoder_mask = self.create_decoder_mask(batch_trg)
        decoder_out = self.decoder(batch_trg, enc_out, encoder_mask, decoder_mask)

        logits = self.fc(decoder_out)
        return logits


    def __beam_search_first_step(self, enc_out, encoder_mask, beam_size, beam_max_len):
        bs, mlen, d_model = enc_out.shape

        batch_trg = torch.full((bs, 1), self.sos_id, device=self.device, dtype=torch.long)
        decoder_mask = self.create_decoder_mask(batch_trg)

        decoder_out = self.decoder(batch_trg, enc_out, encoder_mask, decoder_mask)
        decoder_out = decoder_out[:, -1, :]
        logits = self.fc(decoder_out)

        blocking_list = torch.tensor([
            self.pad_id,
            self.unk_id,
            self.sos_id,
            # self.eos_id
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


    def decoder_batch_beam(self, enc_out, encoder_mask):
        beam_size = self.cfg.beam_size
        bs, enc_mlen, d_model = enc_out.shape
        beam_max_len = enc_mlen + self.cfg.beam_max_length

        scores, full_sequences = self.__beam_search_first_step(enc_out, encoder_mask, beam_size, beam_max_len)

        enc_out = torch.repeat_interleave(enc_out, beam_size, 0)
        encoder_mask = torch.repeat_interleave(encoder_mask, beam_size, 0)

        # blocking_list = torch.tensor([self.pad_id, self.unk_id, self.sos_id], device=self.device, dtype=torch.long)
        for i in range(2, beam_max_len):
            sequences = full_sequences[:, :i]
            decoder_mask = self.create_decoder_mask(sequences)
            decoder_out = self.decoder(sequences, enc_out, encoder_mask, decoder_mask)

            decoder_out = decoder_out[:, -1]
            logits = self.fc(decoder_out)
            # logits[..., blocking_list] = float("-inf")

            log_probs = torch.log_softmax(logits, dim=-1)

            lprob, idx = log_probs.topk(beam_size)
            lprob_rep = torch.full((1, beam_size), float("-inf"), device=self.device)
            lprob_rep[:, 0] = 0
            idx_rep = torch.full((1, beam_size), -1, device=self.device)
            idx_rep[:, 0] = self.eos_id

            check_eos = (full_sequences[:, i-1] == self.eos_id).view(-1, 1)

            lprob = torch.where(check_eos, lprob_rep, lprob)
            idx = torch.where(check_eos, idx_rep, idx)

            # not length norm
            total_scores = (lprob + scores)
            total_scores = total_scores.view(bs, -1)
            top_scores, top_indices = torch.topk(total_scores, beam_size, dim=-1)

            selected_seqs = top_indices // beam_size
            offsets = torch.arange(bs, device=self.device).unsqueeze(1) * beam_size
            selected_seqs = (selected_seqs + offsets).view(-1)

            selected_tokens = top_indices % beam_size
            selected_tokens = selected_tokens.view(-1)

            # new_full_sequences = full_sequences[selected_seqs].clone()
            # new_full_sequences[:, i] = idx[selected_seqs, selected_tokens]
            # full_sequences = new_full_sequences

            full_sequences[:, :len] = full_sequences[selected_seqs, :len]
            full_sequences[:, len] = idx[selected_seqs, selected_tokens]
            scores = top_scores.view(-1, 1)

            check_eos = full_sequences[:, i] == self.eos_id
            if check_eos.all():
                break

        full_sequences = full_sequences.cpu()
        scores = scores.cpu()

        seqs_len = [torch.where(seq == self.eos_id)[0] for seq in full_sequences]
        seqs_len = [int(seq[0]) - 1 if seq.shape[0] > 0 else beam_max_len - 2 for seq in seqs_len]
        seqs_len = torch.tensor(seqs_len).cpu()

        full_sequences = full_sequences.view(bs, beam_size, -1)
        norm_factor = ((seqs_len + 5) / 6) ** self.cfg.length_penalty

        scores = scores.squeeze(1) / norm_factor
        scores = scores.view(bs, beam_size)

        best_idx = scores.argmax(dim=-1)
        best_sequences = full_sequences[torch.arange(bs), best_idx]

        return best_sequences

    def decoder_batch_greedy(self, enc_out, encoder_mask):
        bs, enc_mlen, d_model = enc_out.shape
        max_len = enc_mlen + self.cfg.beam_max_length
        trg_indices = torch.full((bs, max_len), self.pad_id, device=self.device)
        trg_indices[:, 0] = self.sos_id

        finished = torch.zeros(bs, dtype=torch.bool, device=self.device)

        for i in range(max_len - 1):
            trg_input = trg_indices[:, :i+1]
            trg_mask = self.create_decoder_mask(trg_input)

            output = self.decoder(trg_input, enc_out, encoder_mask, trg_mask)

            logits = self.fc(output[:, -1])
            if finished.any():
                logits[finished] = float("-inf")
                logits[finished, self.pad_id] = 0

            next_token = torch.argmax(logits, dim=-1)

            trg_indices[:, i+1] = next_token

            is_eos = next_token == self.eos_id
            finished = finished | is_eos
            if finished.all():
                break

        return trg_indices


    def decoder_greedy(self, enc_out, encoder_mask):
        bs, enc_mlen, d_model = enc_out.shape
        maxlen = enc_mlen + self.cfg.beam_max_length
        res = list()
        for i in range(bs):
            eout = enc_out[i].unsqueeze(0)
            emask = encoder_mask[i].unsqueeze(0)
            seqs = torch.ones((1, 1), device=self.device, dtype=torch.long) * self.sos_id
            for j in range(maxlen - 1):
                dmask = self.create_decoder_mask(seqs)
                out = self.decoder(seqs, eout, emask, dmask)
                out = out[:, -1]
                logits = self.fc(out)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                seqs = torch.concat([seqs, next_token], dim=-1)
                if next_token[0, 0] == self.eos_id:
                    break
            res.append(seqs.squeeze(0))
        res = torch.nn.utils.rnn.pad_sequence(res, padding_value=self.pad_id, batch_first=True)
        return res


    def forward(self, batch_src, trg_teacher=None):
        encoder_mask = self.create_encoder_mask(batch_src)
        enc_out = self.encoder(batch_src, encoder_mask)

        if trg_teacher is not None:
            out = self.decoder_teacher_forcing(trg_teacher, enc_out, encoder_mask)
        else:
            out = self.decoder_batch_beam(enc_out, encoder_mask)
            # out = self.decoder_batch_greedy(enc_out, encoder_mask)
            # out = self.decoder_greedy(enc_out, encoder_mask)
        return out
