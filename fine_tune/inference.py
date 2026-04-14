"""
Inference wrapper for the fine-tuned architectural LLM.
Used by V3 backend to replace MOE generation.

Usage:
    from inference import ArchitectLLM
    llm = ArchitectLLM()
    layout = llm.generate(constraints_dict)

Or standalone test:
    python3 inference.py
"""
import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HERE       = Path(__file__).parent
MODEL_DIR  = HERE / 'final_model'
SYS_PROMPT = (HERE.parent / 'backend/training/prompts/architectural_system_prompt.txt').read_text()

_DEFAULT_CONSTRAINTS = {
    "sqft": 2000, "bedrooms": 3, "bathrooms": 2, "stories": 1,
    "style": "craftsman", "garage": "2car", "laundry": "room",
    "outdoor": "patio", "ceilingHeight": 9,
    "primarySuite": True, "homeOffice": False,
    "formalDining": False, "masterBath": True, "walkInCloset": True,
}


class ArchitectLLM:
    def __init__(self, model_dir: str | Path = MODEL_DIR, quantize: bool = True):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f'Fine-tuned model not found at {model_dir}. '
                'Run fine_tune/train.py first.'
            )

        print(f'Loading architectural LLM from {model_dir}...')
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        if quantize:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                quantization_config=bnb,
                device_map='auto',
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                device_map='auto',
                torch_dtype=torch.bfloat16,
            )
        self.model.eval()
        print('Model loaded.')

    def generate(self, constraints: dict, max_new_tokens: int = 3000) -> dict:
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{SYS_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{json.dumps(constraints)}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return self._parse_layout(raw, constraints)

    def _parse_layout(self, raw: str, constraints: dict) -> dict:
        """Extract JSON from model output, with fallback parsing."""
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Return a minimal error structure
        return {
            'error': 'parse_failed',
            'raw': raw[:500],
            'rooms': [],
            'reasoning': 'Layout generation failed to produce valid JSON.',
        }


if __name__ == '__main__':
    # Quick smoke test
    llm = ArchitectLLM()
    layout = llm.generate(_DEFAULT_CONSTRAINTS)
    print('\n=== Generated Layout ===')
    print(f'Reasoning: {layout.get("reasoning", "N/A")}')
    print(f'Rooms ({len(layout.get("rooms", []))}):')
    for r in layout.get('rooms', []):
        print(f'  {r["name"]:20s} {r["type"]:20s} {r["width"]}×{r["height"]} @ ({r["x"]},{r["y"]})')
    print(f'Footprint: {layout.get("footprint_width")}×{layout.get("footprint_depth")} ft')
    print(f'Sqft: {layout.get("total_conditioned_sqft")}')
