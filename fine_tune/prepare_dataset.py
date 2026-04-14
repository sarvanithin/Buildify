"""
Converts dataset_clean.jsonl → Alpaca-format JSONL ready for LoRA fine-tuning.

Alpaca format:
{
  "instruction": "<system prompt>",
  "input": "<constraint JSON>",
  "output": "<architectural layout JSON>"
}

Usage: python3 prepare_dataset.py
"""
import json
import random
from pathlib import Path

HERE     = Path(__file__).parent
ROOT     = HERE.parent
SYS_PROMPT = (ROOT / 'backend/training/prompts/architectural_system_prompt.txt').read_text()
CLEAN    = ROOT / 'backend/training/data/dataset_clean.jsonl'
OUT_TRAIN = HERE / 'train.jsonl'
OUT_VAL   = HERE / 'val.jsonl'

VAL_RATIO = 0.05  # 5% validation split

def main():
    examples = []
    for line in CLEAN.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        inp    = ex['input']
        output = ex['output']

        alpaca = {
            "instruction": SYS_PROMPT,
            "input": json.dumps(inp),
            "output": json.dumps(output, separators=(',', ':')),
        }
        examples.append(alpaca)

    random.seed(42)
    random.shuffle(examples)

    n_val   = max(50, int(len(examples) * VAL_RATIO))
    val     = examples[:n_val]
    train   = examples[n_val:]

    OUT_TRAIN.parent.mkdir(exist_ok=True)
    with open(OUT_TRAIN, 'w') as f:
        for ex in train:
            f.write(json.dumps(ex) + '\n')

    with open(OUT_VAL, 'w') as f:
        for ex in val:
            f.write(json.dumps(ex) + '\n')

    print(f'Train: {len(train):,} examples → {OUT_TRAIN}')
    print(f'Val:   {len(val):,} examples  → {OUT_VAL}')
    print(f'\nRun fine-tuning with:')
    print(f'  cd fine_tune && pip install -r requirements.txt')
    print(f'  python3 train.py')


if __name__ == '__main__':
    main()
