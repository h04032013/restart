# training with FSDP (multi-GPU ready) 
import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from lightweight_eval_callback import LightweightMathEvalCallback

model_name = "microsoft/Phi-4-mini-instruct"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/7e5_subset6m"
max_length = 1024

# Evaluation settings
MATH_EVAL_FILE = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/MATH/MATH_test.json"
NUM_EVAL_SAMPLES = 4754

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    cache_dir=cache_str,
)
if hasattr(base_model, "config"):
    base_model.config.use_cache = False
    try:
        base_model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

print("Loading data from open-web-math (will cache to disk)...")
dataset = load_dataset(
    "open-web-math/open-web-math",
    split="train[:6000000]", 
    cache_dir=cache_str
)
print(f"Loaded {len(dataset)} samples")

# Shuffle and split
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset, test_dataset = split["train"], split["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

def tokenize(examples):
    return tokenizer(examples["text"], max_length=max_length, truncation=True)

tokenized_train_data = train_dataset.map(tokenize, batched=True)
tokenized_test_data = test_dataset.map(tokenize, batched=True)

tokenized_train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_test_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.001,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

rank = int(os.environ.get("RANK", "0"))
if rank == 0:
    wandb.init(entity="hdiaz-harvard-university", project="training-opwmth")

training_args = TrainingArguments(
    output_dir=ft_cache,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    learning_rate=7e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=1e-6,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=1500,
    save_strategy="steps",
    save_steps=1500,
    logging_steps=100,
    report_to=["wandb"],
    run_name="ft-opwmth-6m",
    save_safetensors=True,

    # ---- FSDP: full shard + auto wrap ----
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "min_num_params": int(5e6),
        "use_orig_params": False,
        "forward_prefetch": True,
        "sync_module_states": True,
    },

    # Dataloader/CPU perf
    dataloader_num_workers=4,  # Can use workers now that it's not streaming!
    dataloader_pin_memory=True,
)

# Create lightweight evaluation callback
try:
    math_eval_callback = LightweightMathEvalCallback(
        eval_data_path=MATH_EVAL_FILE,
        num_samples=NUM_EVAL_SAMPLES,
        summary_file="math_eval_summary.json"
    )
    callbacks = [math_eval_callback]
    print(f"✅ MATH evaluation callback added (evaluating on {NUM_EVAL_SAMPLES} problems)")
except Exception as e:
    print(f"⚠️  Could not load MATH evaluation data: {e}")
    print("    Training will continue without MATH evaluation")
    callbacks = []

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
)

print("WORLD_SIZE=", os.environ.get("WORLD_SIZE"), "RANK=", os.environ.get("RANK"), flush=True)

trainer.train()
trainer.save_model(ft_cache)
print("done training")

