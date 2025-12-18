python test_finetune.py --base_model "unsloth/Qwen2.5-3B-Instruct" \
                        --lora_model "ledas/Qwen2.5-3B-Instruct-LoRA-SFT" \
                        --test_src "./data/vlsp_sft/public_test.vi" \
                        --test_trg "./data/vlsp_sft/public_test.en" \
                        --direction "vi2en" \
                        --batch_size 16
                        # --debug 1
