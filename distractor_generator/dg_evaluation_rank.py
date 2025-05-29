"""
Plausibility evaluation code for distractors.

The baseline distractors must also be in the same format as `distractor_inference_{course}.json`.  
The pairwise ranker is loaded for inference and generates the following output:
"item_no": test set item number, 
"question": question, 
"answer": correct answer,  
"d_list": list of distractors,
"invalid": invalid distractors among the top-3,
"d_reasoning": reasoning results from the pairwise ranker, 
"d_scores": scores from each model
"""

import os
import pandas as pd
import json
import numpy as np
import string
import re
from collections import Counter
import itertools
import torch
import random
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_columns', None)


def has_duplicate_values(dictionary):
    values = list(dictionary.values())
    return len(values) != len(set(values))

def pairwise_rank(model, prompt, question, answer, option_a, option_b):
    score = {'a':0, 'b':0}

    results_ab = []
    results_ba = []

    repeat = 0
    while has_duplicate_values(score):
        repeat += 1
        # AB, BA
        batch_inputs = [prompt.format(question, answer, option_a, option_b), prompt.format(question, answer, option_b, option_a)]
        batch_encoded_inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True, truncation=True).to(device)

        generate_kwargs = dict(
            input_ids=batch_encoded_inputs['input_ids'],
            attention_mask=batch_encoded_inputs['attention_mask'],
            do_sample=True,
            temperature=0.5,  
            max_new_tokens=512, 
            repetition_penalty=1.0
        )

        with torch.no_grad():
            batch_outputs = model.generate(**generate_kwargs)
        batch_decoded_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

        for j, output in enumerate(batch_decoded_outputs):
            # AB
            if j == 0: 
                result = output.split("[/INST]")[1]
                results_ab.append(result)
                if len(result.split("Choice:")) == 2:
                    choice = result.split("Choice:")[1].strip()
                else:
                    choice = "None"
                if 'a' in choice.lower():
                    score['a'] += 1
                elif 'b' in choice.lower():
                    score['b'] += 1

            # BA
            elif j == 1:
                result = output.split("[/INST]")[1]
                results_ba.append(result)
                if len(result.split("Choice:")) == 2:
                    choice = result.split("Choice:")[1].strip()
                else:
                    choice = "None"
                if 'b' in choice.lower():
                    score['a'] += 1
                elif 'a' in choice.lower():
                    score['b'] += 1

        print(score)
        if repeat >= 10:
            break

    max_key = max(score, key=score.get)

    return max_key, score, results_ab, results_ba


prompt = """[INST] You are a teacher analyzing which distractor in a given Multiple Choice Question is more confusing for students and why.
Your review should include the following content in one paragraph:
- Describe realistic process of solving the problem from a student's perspective as you look at each distractor. Consider why it might be plausible as the correct/incorrect statement, based on students' misconceptions, mistakes, intuition, etc., from various angles.
- Output your choice as a single token, either A or B, that students are more likely to choose.

[Question]
{}

[Answer]
{}

[Distractor A]
{}

[Distractor B]
{}

Generate in the following format:
### Review: 
### Choice: [/INST] """

# Load pairwise ranker
# If using an adapter
adapter_name = "adapter_dir" # pairwise ranker adapter directory
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_name,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)
# If using a merged model
# model_name = f"./merged_dir" # pairwise ranker merged model directory
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     #load_in_4bit=True,
# ).to(device)
tokenizer = AutoTokenizer.from_pretrained(adapter_name)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_bos_token = True

course_list = ['python', 'DB', 'MLDL']
doc_list = ['distractor_baseline_{}.json', 'distractor_inference_{}.json'] # List in order: baseline, ours

for course in course_list:
    data = {}

    doc1 = doc_list[0].format(course)
    doc2 = doc_list[1].format(course)
    with open(f"{doc1}", "r", encoding='utf-8-sig') as f:
        doc1_file = json.load(f)    
    with open(f"{doc2}", "r", encoding='utf-8-sig') as f:
        doc2_file = json.load(f)    
    n = len(doc1_file)
    
    for i in range(n):
        if i >= 0 and str(i) not in list(data.keys()):
            idx = str(i)
            print(i)
            item_no = doc1_file[idx]['item_no']
            question = doc1_file[idx]['question']
            answer = doc1_file[idx]['answer']

            d_list = {}
            d_scores = {}
            d_list[doc1] = []
            d_list[doc2] = []
            d_scores[doc1] = 0
            d_scores[doc2] = 0

            list1 = list(doc1_file[idx]['distractors'].keys())
            list2 = list(doc2_file[idx]['distractors'].keys())
            
            for d in range(min(len(list1), len(list2))):
                if len(d_list[doc1]) < 5:
                    d_list[doc1].append(list1[d])
                if len(d_list[doc2]) < 5:
                    d_list[doc2].append(list2[d])
            
            # select unique distractors only
            common_items = set(d_list[doc1]) & set(d_list[doc2])
            unique_list1 = [item for item in d_list[doc1] if item not in common_items]
            unique_list2 = [item for item in d_list[doc2] if item not in common_items]
            
            # top 3
            min_len = 3
            d_list[doc1] = unique_list1[:min_len]
            d_list[doc2] = unique_list2[:min_len]

            print(d_list)
            print(d_scores)

            c1, c2 = d_list.keys() # c1 == A, c2 == B 
            d_reasoning = {}

            for d1i in d_list[c1]:
                for d2i in d_list[c2]:
                    d_reasoning[f"{d1i} _vs_ {d2i}"] = []

                    option_a = d1i
                    option_b = d2i

                    # pairwise rank
                    max_key, score, results_ab, results_ba = pairwise_rank(model, prompt, question, answer, option_a, option_b)

                    if str(max_key).lower() == "a":
                        d_scores[c1] += 1
                    elif str(max_key).lower() == "b":
                        d_scores[c2] += 1
                    print(d_scores)

                    d_reasoning[f"{d1i} _vs_ {d2i}"] = results_ab

                    data[idx] = {"item_no":item_no, "question":question, "answer":answer, "d_list":d_list, "d_reasoning":d_reasoning, "d_scores":d_scores}

                    with open(f'evaluation_rank_{course}_baseline_vs_ours_setting_b.json','w') as f:
                        json.dump(data, f, indent=4)
