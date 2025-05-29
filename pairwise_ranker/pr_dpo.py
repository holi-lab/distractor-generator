"""
DPO training code for the pairwise ranker.

The DPO training set must include the following columns:
"prompt": instruction prompt (question, answer, and distractors must be filled in)
"chosen": correct reasoning and choice
"rejected": incorrect reasoning and choice

Reference: https://github.com/mzbac/llama2-fine-tune
"""

import os
import torch
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DPOConfig, DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from utils import find_all_linear_names, print_trainable_parameters


def return_prompt_and_responses(samples):    
    return {
        "prompt": samples['prompt'],
        "chosen": samples["chosen"],  
        "rejected": samples["rejected"], 
    }

output_dir="output_dir" # output directory
model_name ="merged_SFT_model" # merged SFT model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

train_df = pd.read_csv("pr_dpo_train.csv") # DPO training set
train_dataset = Dataset.from_pandas(train_df[['prompt', 'chosen', 'rejected']]) # target column: 'prompt', 'chosen', 'rejected'

original_columns = train_dataset.column_names

train_dataset.map(
    return_prompt_and_responses,
    remove_columns=original_columns,
)

print(train_dataset[1])
print(len(train_dataset))

training_args = DPOConfig(
    num_train_epochs=5, 
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing =True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    max_grad_norm= 0.3,
    save_steps= 100,
    learning_rate=1e-6,
    bf16=True,
    #save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=True,
    max_prompt_length=1110,
    save_strategy="epoch",
    max_length=1510,  
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

peft_model = get_peft_model(model, peft_config)
print_trainable_parameters(peft_model)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

dpo_trainer.train()
dpo_trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
