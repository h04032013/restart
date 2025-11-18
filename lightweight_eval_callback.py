"""
Lightweight evaluation callback for training
- Evaluates on 50 MATH problems at each checkpoint
- Uses is_equiv for accuracy
- Logs to WandB
- NO large output files saved
"""

import json
import torch
import re
from transformers import TrainerCallback
from pathlib import Path
from math_equivalence import is_equiv
import wandb

class LightweightMathEvalCallback(TrainerCallback):
    """
    Lightweight evaluation on MATH dataset during training.
    Evaluates in-memory, logs to WandB, saves only accuracy summary.
    """
    
    def __init__(self, eval_data_path, num_samples=4754, summary_file="eval_summary.json"):
        """
            eval_data_path: Path to MATH test problems JSON
            num_samples: Number of problems to evaluate on (default: 50)
            summary_file: File to save accuracy history
        """
        self.num_samples = num_samples
        self.summary_file = summary_file
        self.accuracy_history = []
        
        # Load evaluation data
        print(f"\n{'='*80}")
        print(f"Loading MATH evaluation data ({num_samples} problems)...")
        with open(eval_data_path, 'r') as f:
            all_problems = json.load(f)
        
        # Take first num_samples
        self.eval_data = all_problems[:num_samples]
        print(f"✅ Loaded {len(self.eval_data)} problems for evaluation")
        print(f"{'='*80}\n")
    
    def extract_boxed_content(self, latex_str):
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
    
    def extract_answer(self, response):
        """Extract answer from model response"""
        # Try boxed first
        boxed = self.extract_boxed_content(response)
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

    @staticmethod
    def _unwrap_model(model):
        """Handle wrapped models (e.g., DDP/FSDP) by returning the underlying module."""
        return getattr(model, "module", model)
    
    def evaluate_model(self, model, tokenizer):
        """Run evaluation on all problems"""
        model.eval()
        correct = 0
        
        # Prompt template - adjust if needed
        prompt_template = (
            "<|system|>You are providing insightful explanations to grade school students.<|end|>"
            "<|user|>First, break down this problem for grade school students. "
            "Then, clearly state only the final numerical answer in a latex boxed environment which will be scored. \n\n"
            "Problem: {}\n<|end|><|assistant|>"
        )
        
        with torch.no_grad():
            for i, problem_data in enumerate(self.eval_data):
                problem = problem_data['problem']
                correct_answer = problem_data['answer']
                
                # Generate response
                prompt = prompt_template.format(problem)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=612,
                        do_sample=False,  # Greedy decoding for consistency
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    # Decode only new tokens
                    prompt_len = inputs['input_ids'].shape[1]
                    new_tokens = outputs[0, prompt_len:]
                    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Extract answer
                    model_answer = self.extract_answer(response)
                    
                    # Check equivalence
                    if is_equiv(model_answer, correct_answer):
                        correct += 1
                    
                    # Print first 3 for debugging
                    if i < 3:
                        print(f"  Problem {i+1}: {problem[:50]}...")
                        print(f"    Model: {model_answer}")
                        print(f"    Correct: {correct_answer}")
                        print(f"    Match: {'✅' if is_equiv(model_answer, correct_answer) else '❌'}")
                
                except Exception as e:
                    print(f"  Error on problem {i+1}: {e}")
        
        model.train()
        accuracy = correct / len(self.eval_data) if len(self.eval_data) > 0 else 0.0
        return accuracy, correct, len(self.eval_data)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        # Only evaluate on rank 0
        if not state.is_world_process_zero:
            return
        
        print(f"\n{'='*80}")
        print(f"🔍 MATH EVALUATION at step {state.global_step}")
        print(f"{'='*80}")
        
        trainer = getattr(self, "trainer", None)

        model = kwargs.get('model')
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)

        tokenizer = kwargs.get('tokenizer')
        if tokenizer is None and trainer is not None:
            tokenizer = getattr(trainer, "tokenizer", None)
        
        if model is None or tokenizer is None:
            print("⚠️  Warning: Could not get model or tokenizer for evaluation")
            return

        model = self._unwrap_model(model)
        
        # Run evaluation
        accuracy, correct, total = self.evaluate_model(model, tokenizer)
        
        # Save to history
        eval_result = {
            'step': state.global_step,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'epoch': state.epoch
        }
        self.accuracy_history.append(eval_result)
        
        # Log to WandB
        if wandb.run is not None:
            wandb.log({
                'eval/math_accuracy': accuracy,
                'eval/math_correct': correct,
                'eval/math_total': total,
                'step': state.global_step
            })
        
        # Save summary to disk
        summary_path = Path(args.output_dir) / self.summary_file
        with open(summary_path, 'w') as f:
            json.dump(self.accuracy_history, f, indent=2)
        
        print(f"\n📊 Results:")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Summary saved to: {summary_path}")
        print(f"{'='*80}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Print final summary at end of training"""
        if not state.is_world_process_zero:
            return
        
        if not self.accuracy_history:
            return
        
        print(f"\n{'='*80}")
        print("📈 EVALUATION SUMMARY OVER TRAINING")
        print(f"{'='*80}")
        print(f"{'Step':<10} {'Accuracy':<12} {'Correct'}")
        print(f"{'-'*80}")
        for result in self.accuracy_history:
            print(f"{result['step']:<10} {result['accuracy']:<12.2%} {result['correct']}/{result['total']}")
        print(f"{'='*80}\n")


