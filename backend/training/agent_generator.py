"""
Agent-based training data generator.
Reads constraint batches from constraints_all.jsonl and calls the Claude Code
sub-agent harness (run_batch_agent.py) for each batch.

Usage:
    python3 agent_generator.py --start 0 --end 500 --batch-size 10

    --start / --end  : index range into constraints_all.jsonl
    --batch-size     : constraints per agent call (10 is reliable)
    --workers        : parallel agent calls at once (default 5)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).parent
CONSTRAINTS_FILE = HERE / 'constraints_all.jsonl'
DATA_DIR = HERE / 'data'
DATA_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (HERE / 'prompts' / 'architectural_system_prompt.txt').read_text()


def load_constraints(start: int, end: int) -> list[dict]:
    lines = CONSTRAINTS_FILE.read_text().splitlines()
    return [json.loads(l) for l in lines[start:end]]


def build_user_prompt(batch: list[dict]) -> str:
    lines = [
        "Generate architectural floor plan layouts for each of the following constraint sets.",
        "Respond with a JSON array — one object per constraint set, in order.",
        "Each object must have: reasoning, entry_sequence, zones, circulation_notes, solar_notes, rooms, total_conditioned_sqft, footprint_width, footprint_depth, massing_notes.",
        "Follow the system prompt rules exactly. Output ONLY the JSON array, nothing else.",
        "",
        "CONSTRAINT SETS:",
    ]
    for i, c in enumerate(batch):
        lines.append(f"{i+1}. {json.dumps(c)}")
    return '\n'.join(lines)


def run_batch(batch_id: int, constraints: list[dict]) -> dict:
    """Runs one batch and writes output to data/batch_{batch_id:05d}.jsonl"""
    out_file = DATA_DIR / f'batch_{batch_id:05d}.jsonl'
    if out_file.exists():
        print(f'[batch {batch_id:05d}] already done, skipping')
        return {'batch_id': batch_id, 'skipped': True}

    # Write the prompt to a temp file so the sub-process can read it
    prompt_file = DATA_DIR / f'_prompt_{batch_id:05d}.txt'
    prompt_file.write_text(build_user_prompt(constraints))

    # Call claude CLI as a subprocess (uses Max Plan session)
    cmd = [
        'claude', '-p',
        f'{SYSTEM_PROMPT}\n\n{prompt_file.read_text()}',
        '--output-format', 'text',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    prompt_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f'[batch {batch_id:05d}] ERROR: {result.stderr[:200]}')
        return {'batch_id': batch_id, 'error': result.stderr[:200]}

    raw = result.stdout.strip()

    # Parse the JSON array response
    try:
        # Strip markdown code fences if present
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        layouts = json.loads(raw)
        if not isinstance(layouts, list):
            layouts = [layouts]
    except json.JSONDecodeError as e:
        print(f'[batch {batch_id:05d}] JSON parse error: {e}')
        # Save raw for debugging
        (DATA_DIR / f'_raw_{batch_id:05d}.txt').write_text(raw)
        return {'batch_id': batch_id, 'error': str(e)}

    # Write training examples
    written = 0
    with open(out_file, 'w') as f:
        for c, layout in zip(constraints, layouts):
            if not isinstance(layout, dict) or 'rooms' not in layout:
                continue
            example = {
                'input': c,
                'output': layout,
                'source': 'claude_distill_v1',
            }
            f.write(json.dumps(example) + '\n')
            written += 1

    print(f'[batch {batch_id:05d}] wrote {written}/{len(constraints)} examples')
    return {'batch_id': batch_id, 'written': written}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',      type=int, default=0)
    parser.add_argument('--end',        type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--workers',    type=int, default=5)
    args = parser.parse_args()

    all_constraints = load_constraints(args.start, args.end)
    batches = [
        all_constraints[i:i + args.batch_size]
        for i in range(0, len(all_constraints), args.batch_size)
    ]
    batch_ids = list(range(args.start // args.batch_size,
                           args.start // args.batch_size + len(batches)))

    print(f'Generating {len(all_constraints)} layouts in {len(batches)} batches '
          f'({args.workers} parallel workers)...')

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, bid, batch): bid
                   for bid, batch in zip(batch_ids, batches)}
        for future in as_completed(futures):
            results.append(future.result())

    total_written = sum(r.get('written', 0) for r in results)
    errors = [r for r in results if 'error' in r]
    print(f'\nDone. {total_written} examples written. {len(errors)} batch errors.')
    if errors:
        print('Failed batches:', [r['batch_id'] for r in errors])


if __name__ == '__main__':
    main()
