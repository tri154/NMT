from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset
from datasets import DatasetDict
import torch
import os

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
                f"{example['src_lang']} to {example['tgt_lang']}:\n\n"
                f"{example['src']}"
            )
        },
        {
            "role": "assistant",
            "content": example["tgt"]
        }
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    }


def load_parallel_bidirectional(src_path, tgt_path, src_lang, tgt_lang):
    with open(src_path, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f]
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f]

    assert len(src_lines) == len(tgt_lines)

    src_data = []
    tgt_data = []
    src_langs = []
    tgt_langs = []

    for s, t in zip(src_lines, tgt_lines):
        if not s or not t:
            continue

        # src -> tgt
        src_data.append(s)
        tgt_data.append(t)
        src_langs.append(src_lang)
        tgt_langs.append(tgt_lang)

        # tgt -> src
        src_data.append(t)
        tgt_data.append(s)
        src_langs.append(tgt_lang)
        tgt_langs.append(src_lang)

    return Dataset.from_dict({
        "src": src_data,
        "tgt": tgt_data,
        "src_lang": src_langs,
        "tgt_lang": tgt_langs,
    })


def prepare_data():
    TRAIN_EN = "data/vlsp_sft/train.en"
    TRAIN_VI = "data/vlsp_sft/train.vi"
    VALID_EN = "data/vlsp_sft/dev.en"
    VALID_VI = "data/vlsp_sft/dev.vi"
    train_dataset = load_parallel_bidirectional(
        TRAIN_EN, TRAIN_VI,
        src_lang="English",
        tgt_lang="Vietnamese",
    )

    valid_dataset = load_parallel_bidirectional(
        VALID_EN, VALID_VI,
        src_lang="English",
        tgt_lang="Vietnamese",
    )

    train_dataset = train_dataset.map(
        format_mt,
        remove_columns=["src", "tgt", "src_lang", "tgt_lang"]
    )

    valid_dataset = valid_dataset.map(
        format_mt,
        remove_columns=["src", "tgt", "src_lang", "tgt_lang"]
    )

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
    })

    token = os.environ.get("HF_TOKEN")
    dataset_dict.push_to_hub(
        "ledas/vlsp-en-vi-bidirectional-sft",
        private=False,
        token = token
    )


def load_data():
    dataset = load_dataset("ledas/vlsp-en-vi-bidirectional-sft")
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    return train_dataset, valid_dataset


def load_model(r, alpha):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    return model, tokenizer


if __name__ == "__main__":
    OUTPUT_DIR = "./qwen25_mt_lora"
    token = os.environ.get("HF_TOKEN")

    model, tokenizer = load_model(64, 128)
    train_dataset, valid_dataset = load_data()


    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        gradient_accumulation_steps = 4,

        learning_rate = 1e-4,
        num_train_epochs = 1,
        max_steps=15000,
        warmup_steps=100,
        lr_scheduler_type = "cosine",

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),

        logging_steps = 100,
        eval_steps = 1000,
        save_steps = 1000,
        save_total_limit = 2,
        optim = "paged_adamw_8bit",
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        dataset_text_field = "text",
        max_seq_length = 512,
        packing = True,
        args = training_args,
    )

    trainer.train()

    model.push_to_hub(
        "ledas/Qwen2.5-3B-Instruct-LoRA-SFT",
        tokenizer,
        token = token
    )
