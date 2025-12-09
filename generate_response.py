import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import regex as re

#prompt_template = ("<|system|>Solve problems step by step and provide insightful explanations.<|end|><|user|> Explain the following math problem to a grade school student. Finally, on the last line, #provide only the final answer in a latex boxed environment. Problem: {} \n<|end|><|assistant|>")


def extract_boxed_content(latex_str):
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

def generate_response (model_name, input_path, output_path, batch_size):

    
    prompt_template = ("<|system|>You are providing insightful explanations to grade school students.<|end|><|user|>First, break down this problem for grade school students. Then, clearly state the final numerical answer in a latex boxed environment which will be scored. \n\nProblem: {}\n<|end|><|assistant|>")
    
    with open(input_path, "r") as f:
        problems = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    base_model.to(device)
    base_model.eval()

    results =[]

    for i in tqdm(range(0, len(problems), batch_size)):
        batch = problems[i:i+batch_size]
        prompts = [prompt_template.format(entry["problem"]) for entry in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        #print ("generating for: ", i)
        with torch.no_grad():
            outputs = base_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
           # return_dict_in_generate=True,
            )

        input_seq_len = inputs["input_ids"].shape[1]
        for j in range(len(batch)):
            generated_ids = outputs[j][input_seq_len:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            final_answer = extract_boxed_content(response_text)

            results.append({
                "problem":batch[j]["problem"],
                "level": batch[j]["level"],
                "type": batch[j]["type"],
                "solution": response_text,
                "answer": final_answer,
                "unique_id": batch[j]["unique_id"]
        })

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

