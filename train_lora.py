import torch
import random
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from typing import Dict
from transformers import BitsAndBytesConfig



import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["BNB_CUDA_VERSION"] = ""   


DATASET_PATH = "/home/prady/dparse/project/policy_dataset"
MODEL_PATH = "/home/prady/dparse/models/mistral-7b"     # <-- LOCAL MODEL DIR
OUTPUT_DIR = "./mistral_lora_out"
EPOCHS = 7
BATCH_SIZE = 1
ACCUM = 8
LR = 2e-4
MAX_LEN = 1024

# -------------------------------
# LOAD DATASET
# -------------------------------
dataset = load_from_disk(DATASET_PATH)
print("Loaded dataset with rows:", len(dataset))

ds = dataset.train_test_split(test_size=0.05, seed=42)

# -------------------------------
# TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

PROMPT_FMT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

def build_prompt(ins, inp, out):
    return PROMPT_FMT.format(instruction=ins.strip(), input=inp.strip()) + out.strip()

# -------------------------------
# TOKENIZATION
# -------------------------------
def tokenize_fn(batch):
    texts, labels = [], []

    for ins, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        full = build_prompt(ins, inp, out)
        texts.append(full)

    enc = tokenizer(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    # Create labels where prompt tokens = -100
    for i in range(len(texts)):
        prompt = PROMPT_FMT.format(
            instruction=batch["instruction"][i],
            input=batch["input"][i]
        )
        prompt_ids = tokenizer(prompt, truncation=True).input_ids
        plen = len(prompt_ids)

        lab = [-100] * MAX_LEN
        ids = enc["input_ids"][i]

        for j in range(plen, MAX_LEN):
            lab[j] = ids[j]

        labels.append(lab)

    enc["labels"] = labels
    return enc


print("Tokenizing...")
tokenized_ds = ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=ds["train"].column_names
)

# ------------------------------------------
# LOAD MISTRAL LOCALLY (NO QUANTIZATION)
# ------------------------------------------
print("Loading LOCAL model in FP16...")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb,
    device_map={"": 0},         # force single-gpu placement
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False


# ------------------------------------------
# LoRA CONFIG
# ------------------------------------------
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)



model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ------------------------------------------
# TRAINER
# ------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,  # REQUIRED for 6GB VRAM
    optim="adamw_torch",          # <-- FIX: works on Windows, no bitsandbytes
    no_cuda=False,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    data_collator=data_collator
)

# ------------------------------------------
# TRAIN
# ------------------------------------------
print("Training begins...")
trainer.train()

model.save_pretrained(OUTPUT_DIR + "/adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/adapter")

print("Training complete. Adapter saved.")
