from unsloth import FastLanguageModel
from datasets import Dataset
from datasets import load_dataset
from datasets import DatasetDict
import sacrebleu
import torch
import os
import numpy as np
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

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
    ]

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
        "reference": example["tgt"],
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
    TRAIN_EN = "data/vlsp_grpo/train.en"
    TRAIN_VI = "data/vlsp_grpo/train.vi"
    VALID_EN = "data/vlsp_grpo/dev.en"
    VALID_VI = "data/vlsp_grpo/dev.vi"

    train_dataset = load_parallel_bidirectional(
        TRAIN_EN, TRAIN_VI,
        src_lang="English",
        tgt_lang="Vietnamese",
    )

    # valid_dataset = load_parallel_bidirectional(
    #     VALID_EN, VALID_VI,
    #     src_lang="English",
    #     tgt_lang="Vietnamese",
    # )

    train_dataset = train_dataset.map(
        format_mt,
        remove_columns=["src", "tgt", "src_lang", "tgt_lang"]
    )

    # valid_dataset = valid_dataset.map(
    #     format_mt,
    #     remove_columns=["src", "tgt", "src_lang", "tgt_lang"]
    # )

    dataset_dict = DatasetDict({
        "train": train_dataset,
        # "validation": valid_dataset,
    })

    token = os.environ.get("HF_TOKEN")
    dataset_dict.push_to_hub(
        "ledas/vlsp-en-vi-bidirectional-grpo",
        private=False,
        token = token
    )


def mt_reward_fn(prompts, completions, reference, **kwargs):
    rewards = []

    for pred, ref in zip(completions, reference):
        pred = pred.strip()
        ref = ref.strip()

        if len(pred) == 0:
            rewards.append(-1.0)
            continue

        bleu = sacrebleu.sentence_bleu(
            pred,
            [ref],
            smooth_method="exp",
        ).score

        chrf = sacrebleu.sentence_chrf(
            pred,
            [ref],
            word_order=2,
        ).score

        bleu /= 100.0
        chrf /= 100.0

        reward = 0.7 * bleu + 0.3 * chrf
        rewards.append(reward)

    return rewards

def load_model(r, alpha):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True
    )
    model.load_adapter(
        "ledas/Qwen2.5-3B-Instruct-LoRA-SFT",
        adapter_name="sft_custom",
    )
    model.set_adapter("sft_custom")


    model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    for name, param in model.named_parameters():
        if "sft_custom" in name:
            param.requires_grad_(False)

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            assert "default" in name

    print(model.peft_config.keys())
    return model, tokenizer

def load_data():
    dataset = load_dataset("ledas/vlsp-en-vi-bidirectional-grpo")
    train_dataset = dataset["train"]

    return train_dataset

def filter_data(train_dataset):
    tokenized = train_dataset.map(
        lambda x: {
            "tokens": tokenizer(
                x["text"],
                add_special_tokens=False,
            )["input_ids"]
        }
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], 0.99))
    print("Max prompt length (p90):", maximum_length)
    dataset = train_dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    dataset = dataset.rename_column("text", "prompt")
    return dataset


if __name__ == "__main__":
    OUTPUT_DIR = "./qwen25_mt_lora_grpo"
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise Exception("not token")

    train_dataset = load_data()
    train_dataset, max_prompt_length = filter_data(train_dataset)
    model, tokenizer = load_model(32, 32)



    model.push_to_hub(
        "ledas/Qwen2.5-3B-Instruct-LoRA-GRPO",
        adapter_name="default",
        token=token,
    )

    max_completion_length = 1024 - max_prompt_length

    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )


    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 100,
        save_steps = 100,
        report_to = "none", # Can use Weights & Biases
        output_dir = OUTPUT_DIR,
    )
