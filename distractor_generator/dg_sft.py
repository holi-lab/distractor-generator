"""
SFT training code for the distractor generator.

The SFT training set must include the following column:
"text": instruction prompt (question and answer must be filled in) + type + n distractors

Reference: https://github.com/mzbac/llama2-fine-tune
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from utils import find_all_linear_names, print_trainable_parameters
from torch.utils.data import DataLoader, RandomSampler
import itertools


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        output_texts.append(example['text'][i])
    return output_texts


def preview_batch(data_loader, tokenizer, num_batches=1):
    for i, batch in enumerate(itertools.islice(data_loader, num_batches)):
        print(f"Batch {i+1}:")
        for j in range(len(batch['text'])):
            print(f"Sample {j+1}: {batch['text'][j]}")
        print("\n")
        if i >= num_batches - 1:
            break


output_dir="output_dir" # output directory
model_name ="pretrained_model" # Mistral-7B-Instruct_v0.2

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token   # https://huggingface.co/docs/trl/sft_trainer  Make sure to have a pad_token_id which is different from eos_token_id which can result in the model not properly predicting EOS (End of Sentence) tokens during generation.
tokenizer.padding_side = "right" 

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

base_model = get_peft_model(base_model, peft_config)

train_df = pd.read_csv(f"dg_sft_train.csv") # SFT training set
train_dataset = Dataset.from_pandas(train_df[['text']])
print(f"Train dataset size: {len(train_dataset)}")

print(train_dataset.shape)
print(train_dataset[11])

response_template = "[/INST]"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    max_grad_norm=0.3,
    learning_rate=2e-4,
    bf16=True,
    #save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="epoch",
    #evaluation_strategy="epoch"
)

trainer = SFTTrainer(
    base_model,
    peft_config=peft_config,
    train_dataset=train_dataset,
    #eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_seq_length=1267,
    formatting_func=formatting_prompts_func,
    args=training_args,
    data_collator=collator
)

data_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=training_args.per_device_train_batch_size
)

preview_batch(data_loader, tokenizer)

trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
