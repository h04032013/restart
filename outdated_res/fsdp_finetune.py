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

model_name = "microsoft/Phi-4-mini-instruct"
#model_name="/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/phi4"
cache_str = "/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
ft_cache = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/lr_7e5_mgn_neweval"
max_length = 1024

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    cache_dir=cache_str,
)
# Disable cache during training for correctness with gradient checkpointing/FSDP
if hasattr(base_model, "config"):
    base_model.config.use_cache = False
    # If flash-attn is installed, this helps memory/speed:
    try:
        base_model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

dataset = load_dataset("open-web-math/open-web-math", cache_dir=cache_str)
dataset = dataset["train"].shuffle(seed=42)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset, test_dataset = split["train"], split["test"]

print(f"Subset size: {len(dataset)}")
print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

def tokenize(examples):
    return tokenizer(examples["text"], max_length=max_length, truncation=True)

tokenized_train_data = train_dataset.map(tokenize, batched=True)
tokenized_test_data = test_dataset.map(tokenize, batched=True)

tokenized_train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_test_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

print("TRAIN[0] SAMPLE", tokenized_train_data[0])
print("TEST[0] SAMPLE", tokenized_test_data[0])

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
    per_device_train_batch_size=1,           # keep micro-batch tiny
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,          # change to 8
    num_train_epochs=1,
    learning_rate=7e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.2,
    weight_decay=1e-6,
    max_grad_norm=1.0,
    gradient_checkpointing=True, 
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=1500,
    logging_steps=100, #10 steps
    report_to=["wandb"],
    run_name="ft-opwmth",
    save_safetensors=True,

    # ---- FSDP: full shard + auto wrap ----
    # "auto_wrap" uses a policy so you don't have to hardcode layer class names.
    fsdp="full_shard auto_wrap",
    fsdp_config={
        # Shard everything, but only auto-wrap large modules to reduce overhead:
        "min_num_params": int(5e6),
        # With PEFT/LoRA, this avoids parameter aliasing issues:
        "use_orig_params": False,
        # Performance knobs:
        "forward_prefetch": True,
        "sync_module_states": True,  # broadcast weights at start
        # If VRAM is very tight, consider CPU offload (slow):
        # "fsdp_offload_params": True,
    },

    # Dataloader/CPU perf
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("WORLD_SIZE=", os.environ.get("WORLD_SIZE"), "RANK=", os.environ.get("RANK"), flush=True)

trainer.train()
trainer.save_model(ft_cache)
print("done training")
'''
from pathlib import Path
import shutil

from peft import PeftModel
from peft.utils import get_peft_model_state_dict

# ---- Train once ----
trainer.train()
print("done training")

# ---- Paths ----
tmp_adapters = Path(ft_cache) / "_tmp_adapters"
merged_dir   = Path(ft_cache) / "merged_model2"
tmp_adapters.mkdir(parents=True, exist_ok=True)
merged_dir.mkdir(parents=True, exist_ok=True)

if trainer.is_world_process_zero():  # only rank 0 does the IO
    model_wrapped = trainer.model

    # 1) Grab LoRA adapter weights from the FSDP-wrapped model onto CPU (rank-0 full state dict)
    peft_sd = None
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(
            model_wrapped,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            peft_sd = get_peft_model_state_dict(model_wrapped)
    except Exception:
        # Non-FSDP or older torch: fall back to a regular state dict grab
        peft_sd = get_peft_model_state_dict(model_wrapped)

    # 2) Save adapters to a temporary dir (writes adapter_model.safetensors + adapter_config.json)
    #    Passing state_dict=... avoids any FSDP naming issues.
    model_wrapped.save_pretrained(str(tmp_adapters), state_dict=peft_sd)

    # 3) Rebuild a clean, non-FSDP base model and load the adapters
    base_cpu = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, cache_dir=cache_str
    )
    peft_loaded = PeftModel.from_pretrained(base_cpu, str(tmp_adapters))

    # 4) Merge LoRA into the base weights and save a standalone model + tokenizer
    merged = peft_loaded.merge_and_unload()
    try:
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
    except TypeError:
        merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # 5) Clean up
    shutil.rmtree(tmp_adapters, ignore_errors=True)
    print(f"[saved] merged model + tokenizer → {merged_dir}")
'''