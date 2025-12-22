from unsloth import FastLanguageModel
from datasets import Dataset
import argparse
import torch
from tqdm import tqdm
import numpy as np
import sacrebleu

def load_data(src_path, trg_path, src_lang, trg_lang, tokenizer, tokenize=False):
    def format_mt(example):
        messages = [
            {
                "role": "system",
                "content": "You are a professional machine translation system."
            },
            {
                "role": "user",
                "content": (
                    f"Translate the following sentence from "
                    f"{src_lang} to {trg_lang}:\n\n"
                    f"{example['src']}"
                )
            },
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True,
            )
        }
    with open(src_path, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f]
    with open(trg_path, encoding="utf-8") as f:
        trg_lines = [l.strip() for l in f]
    assert len(src_lines) == len(trg_lines)

    src_data = []
    trg_data = []
    for s, t in zip(src_lines, trg_lines):
        if not s or not t:
            continue

        src_data.append(s)
        trg_data.append(t)

    dataset = Dataset.from_dict({
        "src": src_data,
    })
    dataset = dataset.map(
        format_mt,
        remove_columns=["src"]
    )
    return dataset, trg_data

def run_inference(dataset, tokenizer, model, max_new_tokens, batch_size, debug=None):
    predictions = []
    BATCH_SIZE = batch_size
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i : i + BATCH_SIZE]

        texts = batch["text"]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(texts)):
            input_len = inputs["input_ids"][j].shape[-1]
            decoded = tokenizer.decode(
                outputs[j][input_len:],
                skip_special_tokens=True,
            ).strip()
            if "<|assistant|>" in decoded:
                decoded = decoded.split("<|assistant|>")[-1].strip()
            predictions.append(decoded)
        if debug is not None:
            return predictions
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--test_src", type=str)
    parser.add_argument("--test_trg", type=str)
    parser.add_argument("--direction", type=str, help="Direction, vi2en or en2vi.")
    parser.add_argument("--batch_size", type=int, default=16, help="Direction, vi2en or en2vi.")
    parser.add_argument("--debug", type=int, default=None)

    args = parser.parse_args()
    debug = args.debug
    src_path = args.test_src
    trg_path = args.test_trg
    batch_size = args.batch_size
    if args.direction == "vi2en":
        src_lang = "Vietnamese"
        trg_lang = "English"
    elif args.direction == "en2vi":
        src_lang = "English"
        trg_lang = "Vietnamese"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = 1024,
        dtype=None,
        load_in_4bit = True,
    )
    if args.lora_model is not None:
        model.load_adapter(args.lora_model)

    FastLanguageModel.for_inference(model)

    dataset, labels = load_data(src_path, trg_path, src_lang, trg_lang, tokenizer, False)
    tgt_token_lens = []
    for trg in labels:
        tokens = tokenizer(
            trg,
            add_special_tokens=False,
        )["input_ids"]
        tgt_token_lens.append(len(tokens))

    lens = np.array(tgt_token_lens)
    max_new_tokens = int(np.percentile(lens, 99)) + 10
    print(f"Inference with max_new_tokens = {max_new_tokens} .")

    predictions = run_inference(dataset, tokenizer, model, max_new_tokens, batch_size, debug)
    if debug is not None:
        labels=labels[:batch_size]
    score = sacrebleu.corpus_bleu(predictions, [labels]).score
    print(f"BLEU score: {score}")
