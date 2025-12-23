This repository contains code of two small projects:
- Transformer from scratch.
- Solution for VLSP 2025 Shared Task on Medical Machine Translation.

## Tehcnical report
[pdf](https://drive.google.com/file/d/12_QB8FJ5feBFCWuJTuHRDSy9sEPtm5DB/view?usp=sharing)

## Variants
This branch contains the code of Transformer version **Bidirect + punc** as described in the report. For other versions, checkout:
| Variants              | Commit/Branch | Notebook |
|----------------------|-------|--------------------|
| Word Tokenization    | [git](https://github.com/tri154/NMT/tree/ls_loss) | [notebook](https://www.kaggle.com/code/saverysad/torch-text-nmt?scriptVersionId=286292791)|
| BPE                  | [git](https://github.com/tri154/NMT/tree/bpe) | [notebook](https://www.kaggle.com/code/saverysad/torch-bpe/log?scriptVersionId=286344760)|
| BPE + Weight tying   | [git](https://github.com/tri154/NMT/tree/ace322741bf7) |[notebook](https://www.kaggle.com/code/nguynduyanh2004/nmt-weight-tying?scriptVersionId=286407670) |
| BPE + Weight tying + Shared vocab   | [git](https://github.com/tri154/NMT/tree/5df340a066eb) |[notebook](https://www.kaggle.com/code/qwerty197/nmt-bidirect-rerun?scriptVersionId=287269647) |
| Modern  | [git](https://github.com/tri154/NMT/tree/1da8247c88d9) |[notebook](https://www.kaggle.com/code/saverysad/imporved-nmt?scriptVersionId=287782478) |
| Modern + bidirect  | [git](https://github.com/tri154/NMT/tree/1da8247c88d9) |[notebook](https://www.kaggle.com/code/trikaggle/imporved-nmt?scriptVersionId=287799505) |
| Birdirect + no_punc  | [git](https://github.com/tri154/NMT/tree/4d01426c9a1a) |[notebook](https://www.kaggle.com/code/saverysad/nmt-bidirect-actual-rerun?scriptVersionId=287781906) |
| Birdirect + punc  | [git](https://github.com/tri154/NMT) |[notebook](https://www.kaggle.com/code/saverysad/nmt-bidirect-actual-rerun?scriptVersionId=287916769) |

## Environment setup
For Transformer from scratch, only requires [pytorch >= 2.8.0](https://pytorch.org/get-started/previous-versions/#:~:text=org/whl/cpu-,v2.8.0,-Wheel) and:
#### `requirements.txt`
```python
sacrebleu # BLEU score
sentencepiece # BPE tokenizer
```

For SFT + GRPO finetuning:
#### `requirements.txt`
```python
sacrebleu
unsloth
vllm
```
Or follow installation from [official unsloth notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks).

## Training Transformer from scratch
To train Transformer from scratch, modify `configs/config.yaml` file (default config file in `main.py`).
Other variants need their specific config file.


#### `configs/config.yaml`
```yaml
device: cuda
# folders
result_path: results/iwslt # saved model and tokenizer
data_dir: data/iwslt_en_vi # what dataset to train

# preprocessing
num_examples: -1
train_max_len: 50
lowercase: false

# tokenizer
tkn_type: bpe # bpe or word
# word tokenizer
min_freq: 5
# bpe tokenizer
tkn_prefix: spm_tkn
vocab_size: 15000

# training
num_epochs: 20
train_batch_size: 128
num_warmups: 4000 # steps
eval_freq: -1 # every $ batchs
print_freq: 200
bidirect: true # bidirectional training
#loss
label_smoothing: 0.1
# optimizer
lr: !!float 0.2 # scheduler detemines lr
opt_b1: 0.9
opt_b2: 0.98
opt_eps: !!float 1e-9

# inference
test_batch_size: 16
#beam
beam_size: 5
beam_max_length: 50
length_penalty: 0.6

# model
dropout: 0.1
d_model: 512
d_ff: 2048
n_heads: 8
pe_max_seq_len: 200
pre_norm: true
n_encoder_layers: 6
n_decoder_layers: 6
```
Run training:

```bash
python main.py
```
The test score will be printed when training is done.
## Finetuning for VLSP task
### SFT stage:
```bash
HF_TOKEN=<your_huggingface_token> python finetune_sft.py
```
Make sure that the pushed model name is correctly set in the `finetune_sft.py`.

### GRPO stage:
```bash
HF_TOKEN=<your_huggingface_token> python finetune_grpo.py
```

Make sure that the adapter name from the SFT stage and the pushed model name are correctly set in `finetune_grpo.py`

### Evaluation
Pretrained adapters of two stages:
- SFT: [ledas/Qwen2.5-3B-Instruct-LoRA-SFT](https://huggingface.co/ledas/Qwen2.5-3B-Instruct-LoRA-SFT)
- GRPO: [ledas/Qwen2.5-3B-Instruct-LoRA-GRPO-5000](https://huggingface.co/ledas/Qwen2.5-3B-Instruct-LoRA-GRPO-5000)

To evaluate the model on public test set:
##### English to Vietnamese
```bash
python test_finetune_sft_grpo.py --base_model "unsloth/Qwen2.5-3B-Instruct" \
                                  --sft_model "ledas/Qwen2.5-3B-Instruct-LoRA-SFT" \
                                  --grpo_model "ledas/Qwen2.5-3B-Instruct-LoRA-GRPO-5000" \
                                  --test_src "./data/vlsp_sft/public_test.en" \
                                  --test_trg "./data/vlsp_sft/public_test.vi" \
                                  --direction "en2vi" \
                                  --batch_size 16
```

##### Vietnamese to English
```bash
python test_finetune_sft_grpo.py --base_model "unsloth/Qwen2.5-3B-Instruct" \
                                  --sft_model "ledas/Qwen2.5-3B-Instruct-LoRA-SFT" \
                                  --grpo_model "ledas/Qwen2.5-3B-Instruct-LoRA-GRPO-5000" \
                                  --test_src "./data/vlsp_sft/public_test.vi" \
                                  --test_trg "./data/vlsp_sft/public_test.en" \
                                  --direction "vi2en" \
                                  --batch_size 16
```

To evaluate on custom test set, simply change `--test_src` and `--test_trg ` to the files. **Make sure `--direction` is set correctly.**

Example evaluation notebook: [Kaggle](https://www.kaggle.com/code/nguynduyanh2004/grpo-nmt-nmt?scriptVersionId=287920371)

## Email
22028165@vnu.edu.vn
