"""
LoRA fine-tuning for architectural floor plan generation.
Base model: meta-llama/Meta-Llama-3.1-8B-Instruct
Method: QLoRA (4-bit quantization + LoRA adapters)
Hardware: single A100 40GB (RunPod / Lambda Labs)

Usage:
    pip install -r requirements.txt
    python3 train.py

Outputs:
    ./checkpoints/        — best checkpoint per epoch
    ./final_model/        — merged model ready for inference
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

HERE = Path(__file__).parent
TRAIN_FILE = HERE / 'train.jsonl'
VAL_FILE   = HERE / 'val.jsonl'

MODEL_ID  = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
MAX_SEQ   = 4096   # architectural layouts can be ~2000 tokens
BATCH     = 2
GRAD_ACCUM = 8     # effective batch = 16
EPOCHS    = 3
LR        = 2e-4
LORA_R    = 64
LORA_ALPHA = 128

SYS_PROMPT = (HERE.parent / 'backend/training/prompts/architectural_system_prompt.txt').read_text()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def format_example(ex: dict) -> str:
    """Format as LLaMA3 chat template."""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYS_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{ex['input']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{ex['output']}<|eot_id|>"
    )


def main():
    print(f'Loading dataset...')
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)
    print(f'  Train: {len(train_raw):,}  Val: {len(val_raw):,}')

    train_ds = Dataset.from_list([{'text': format_example(e)} for e in train_raw])
    val_ds   = Dataset.from_list([{'text': format_example(e)} for e in val_raw])

    # --- 4-bit quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f'Loading base model {MODEL_ID}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias='none',
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=str(HERE / 'checkpoints'),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        optim='paged_adamw_8bit',
        learning_rate=LR,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        weight_decay=0.001,
        evaluation_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=10,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        report_to='tensorboard',
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field='text',
        max_seq_length=MAX_SEQ,
        packing=True,  # pack multiple short examples → faster training
    )

    print('Starting training...')
    trainer.train()

    # --- Merge LoRA into base and save ---
    print('Merging LoRA adapters...')
    merged = trainer.model.merge_and_unload()
    final_dir = HERE / 'final_model'
    merged.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f'Final model saved to {final_dir}')


if __name__ == '__main__':
    main()
