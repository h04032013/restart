"""
Evaluate existing checkpoints using the same logic as lightweight_eval_callback
Works with both adapter checkpoints and merged models

Usage:
    # Evaluate an adapter checkpoint
    python eval_existing_checkpoint.py --checkpoint ./hgf_new_hub/lr_7e5_mgn_neweval/checkpoint-3000 --is_adapter

    # Evaluate a merged model
    python eval_existing_checkpoint.py --checkpoint ./hgf_new_hub/lr_7e5_mgn_neweval/merged_model

    # Evaluate on more problems
    python eval_existing_checkpoint.py --checkpoint ./path/to/checkpoint --is_adapter --num_samples 100
"""

import argparse
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from math_equivalence import is_equiv
from tqdm import tqdm

def extract_boxed_content(latex_str):
    """Extract answer from \\boxed{...} format"""
    final_answer = []
    i = 0
    while i < len(latex_str):
        if latex_str.startswith(r'\boxed{', i):
            i += len(r'\boxed{')
            brace_depth = 1
            content_start = i
            while i < len(latex_str) and brace_depth > 0:
                if latex_str[i] == '{':
                    brace_depth += 1
                elif latex_str[i] == '}':
                    brace_depth -= 1
                i += 1
            if brace_depth == 0:
                content = latex_str[content_start:i - 1]
                final_answer.append(content)
        else:
            i += 1
    return final_answer[0] if final_answer else None

def extract_answer(response):
    """Extract answer from model response"""
    # Try boxed first
    boxed = extract_boxed_content(response)
    if boxed:
        return boxed
    
    # Try "The answer is:" pattern
    match = re.search(r'[Tt]he answer is:?\s*([^\n\.]+)', response)
    if match:
        return match.group(1).strip()
    
    # Return last line
    lines = response.strip().split('\n')
    if lines:
        return lines[-1].strip()
    
    return response.strip()

def evaluate_checkpoint(model, tokenizer, eval_data, verbose=False):
    """
    Evaluate model on MATH problems
    Uses same logic as lightweight_eval_callback
    """
    model.eval()
    correct = 0
    results = []
    
    # Prompt template - same as in callback
    prompt_template = (
        "<|system|>You are providing insightful explanations to grade school students.<|end|>"
        "<|user|>First, break down this problem for grade school students. "
        "Then, clearly state only the final numerical answer in a latex boxed environment which will be scored. \n\n"
        "Problem: {}\n<|end|><|assistant|>"
    )
    
    print(f"\nEvaluating on {len(eval_data)} problems...")
    
    with torch.no_grad():
        for i, problem_data in enumerate(tqdm(eval_data, desc="Evaluating")):
            problem = problem_data['problem']
            correct_answer = problem_data['answer']
            
            # Generate response
            prompt = prompt_template.format(problem)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=612,  # Same as callback
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Decode only new tokens
                prompt_len = inputs['input_ids'].shape[1]
                new_tokens = outputs[0, prompt_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Extract answer
                model_answer = extract_answer(response)
                
                # Check equivalence
                is_correct = is_equiv(model_answer, correct_answer)
                if is_correct:
                    correct += 1
                
                # Store result
                results.append({
                    'problem_id': i,
                    'problem': problem[:100] + "..." if len(problem) > 100 else problem,
                    'model_answer': model_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct
                })
                
                # Print first few examples
                if verbose and i < 5:
                    print(f"\n{'='*80}")
                    print(f"Problem {i+1}: {problem[:80]}...")
                    print(f"  Model answer: {model_answer}")
                    print(f"  Correct answer: {correct_answer}")
                    print(f"  Match: {'✅' if is_correct else '❌'}")
                
            except Exception as e:
                print(f"  Error on problem {i}: {e}")
                results.append({
                    'problem_id': i,
                    'error': str(e)
                })
    
    accuracy = correct / len(eval_data) if len(eval_data) > 0 else 0.0
    return accuracy, correct, len(eval_data), results

def main():
    parser = argparse.ArgumentParser(description="Evaluate existing checkpoint on MATH dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint (adapter or merged model)")
    parser.add_argument("--is_adapter", action="store_true",
                       help="Set if checkpoint is LoRA adapter (not merged)")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-4-mini-instruct",
                       help="Base model name (only needed for adapters)")
    parser.add_argument("--math_file", type=str, 
                       default="/n/netscratch/dam_lab/Lab/hdiaz/ft_project/MATH/MATH_test.json",
                       help="Path to MATH test problems JSON")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of problems to evaluate on")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Optional: Save detailed results to JSON")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output for first 5 problems")
    parser.add_argument("--cache_dir", type=str, 
                       default="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub",
                       help="HuggingFace cache directory")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CHECKPOINT EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Adapter mode: {args.is_adapter}")
    print(f"MATH file: {args.math_file}")
    print(f"Num samples: {args.num_samples}")
    print("="*80 + "\n")
    
    # Load evaluation data
    print("Loading MATH evaluation data...")
    with open(args.math_file, 'r') as f:
        all_problems = json.load(f)
    
    eval_data = all_problems[:args.num_samples]
    print(f"✅ Loaded {len(eval_data)} problems\n")
    
    # Load model
    print("Loading model...")
    if args.is_adapter:
        print(f"  Loading base model: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        print(f"  Loading adapter from: {args.checkpoint}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    else:
        print(f"  Loading merged model from: {args.checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, cache_dir=args.cache_dir)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Model loaded\n")
    
    # Run evaluation
    accuracy, correct, total, results = evaluate_checkpoint(
        model, 
        tokenizer, 
        eval_data,
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*80 + "\n")
    
    # Save detailed results if requested
    if args.output_file:
        output_data = {
            'checkpoint': args.checkpoint,
            'is_adapter': args.is_adapter,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Detailed results saved to: {args.output_file}\n")

if __name__ == "__main__":
    main()

