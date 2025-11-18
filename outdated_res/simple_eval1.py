"""
Dead simple checkpoint evaluation - no frills, just accuracy
Usage: python simple_eval.py checkpoint_path [--adapter] [--n NUM] [--batch-size B]
"""

import sys
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_equivalence import is_equiv
from tqdm import tqdm

# Same functions from callback
def extract_boxed(text):
    i = 0
    while i < len(text):
        if text.startswith(r'\boxed{', i):
            i += 7
            depth = 1
            start = i
            while i < len(text) and depth > 0:
                if text[i] == '{': depth += 1
                elif text[i] == '}': depth -= 1
                i += 1
            if depth == 0:
                return text[start:i-1]
        i += 1
    return None

def extract_answer(text):
    boxed = extract_boxed(text)
    if boxed: return boxed
    match = re.search(r'[Tt]he answer is:?\s*([^\n\.]+)', text)
    if match: return match.group(1).strip()
    return text.strip().split('\n')[-1].strip() if text.strip() else ""



# Load data
with open('/n/netscratch/dam_lab/Lab/hdiaz/ft_project/MATH/MATH_test.json') as f:
    all_problems = json.load(f)

if len(sys.argv) < 2:
    raise SystemExit("Usage: python simple_eval.py CHECKPOINT_PATH [--adapter] [--n NUM] [--batch-size B]")

ckpt = sys.argv[1]

is_adapter = False
requested_n = None
batch_size = 1

args = sys.argv[2:]
i = 0
while i < len(args):
    arg = args[i]
    if arg == "--adapter":
        is_adapter = True
        i += 1
    elif arg == "--n":
        if i + 1 >= len(args):
            raise SystemExit("--n flag requires an integer value")
        requested_n = int(args[i + 1])
        i += 2
    elif arg == "--batch-size":
        if i + 1 >= len(args):
            raise SystemExit("--batch-size flag requires an integer value")
        batch_size = max(1, int(args[i + 1]))
        i += 2
    else:
        raise SystemExit(f"Unrecognized argument: {arg}")

total_available = len(all_problems)
num = total_available if requested_n is None else min(requested_n, total_available)
problems = all_problems[:num]

print(f"Checkpoint: {ckpt}")
print(f"Adapter: {is_adapter}, Samples: {num} (of {total_available}), Batch size: {batch_size}\n")

# Load model
print("Loading model...")
if is_adapter:
    base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct", 
        torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub")
    model = PeftModel.from_pretrained(base, ckpt)
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct",
        cache_dir="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub")
else:
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16,
        device_map="auto", cache_dir="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub")
    tok = AutoTokenizer.from_pretrained(ckpt, cache_dir="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub")

tok.pad_token = tok.eos_token
model.eval()

# Evaluate
prompt_template = "<|system|>You are providing insightful explanations to grade school students.<|end|><|user|>First, break down this problem for grade school students. Then, clearly state only the final numerical answer in a latex boxed environment which will be scored. \n\nProblem: {}\n<|end|><|assistant|>"

correct = 0
with torch.no_grad():
    total_batches = (len(problems) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(problems), batch_size), total=total_batches):
        batch = problems[start:start + batch_size]
        prompts = [prompt_template.format(p['problem']) for p in batch]

        inputs = tok(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=612,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True
        )

        prompt_lengths = inputs['attention_mask'].sum(dim=1)

        for i, problem in enumerate(batch):
            prompt_len = prompt_lengths[i].item()
            new_tokens = outputs[i, prompt_len:]
            response = tok.decode(new_tokens, skip_special_tokens=True)
            model_ans = extract_answer(response)

            if is_equiv(model_ans, problem['answer']):
                correct += 1

print(f"\nAccuracy: {correct}/{num} = {100*correct/num:.1f}%")

