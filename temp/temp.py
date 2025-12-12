
    t_score  = tester.test(model, tokenizer, tag='test', batch_size=cfg.test_batch_size)
    print(t_score)
    # ==================



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
            after_enc_out = enc_out[flat_beam_idx]
            assert (after_enc_out == enc_out).all()
            enc_out = after_enc_out
            if encoder_mask is not None:
                after_encoder_mask = encoder_mask[flat_beam_idx]
                assert (after_encoder_mask == encoder_mask).all()
                encoder_mask = after_encoder_mask

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
