"""
Inference code for the pairwise ranker.

The test set csv must include the following columns:
"item_no": test set item number, 
"question": question, 
"answer": correct answer, 
"option_a": distractor A,
"option_b": distractor B,
"chosen": the distractor with the higher actual selection rate,
"rejected": the distractor with the lower actual selection rate

The output JSON file is structured as follows:
"item_no": test set item number, 
"question": question, 
"answer": correct answer, 
"A": distractor A, 
"B": distractor B, 
"review_ab": reasoning results for AB input (list), 
"review_ba": reasoning results for BA input (list), 
"score": scores for A and B after repeating until consistency is achieved, 
"choice": the final choice made by the model (the one with the higher score), 
"true": the distractor with the higher actual selection rate (used to check model accuracy)
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def has_duplicate_values(dictionary):
    values = list(dictionary.values())
    return len(values) != len(set(values))

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
### Choice: [/INST]"""

# If using an adapter
adapter_name = "adapter_dir" # adapter directory
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_name,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)
# If using a merged model
# model_name = f"./merged_dir" # merged model directory
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
# ).to(device)
tokenizer = AutoTokenizer.from_pretrained(adapter_name)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_bos_token = True

data = {}
course_list = ["python", "DB", "MLDL"]

for course in course_list: 
    df = pd.read_csv(f'test_{course}.csv')  # Test set for each subject
    df = df.iloc[::2] # Assume the n-th row of a pair contains AB and n+1 contains BA. Use only the first row for a pair.
    df.reset_index(drop=True, inplace=True)
    count = 0

    for i in range(len(df)):
        if i >= 0:
            item_no = df['item_no'][i]
            question = df['question'][i]
            answer = df['answer'][i]
            option_a = df['option_a'][i]
            option_b = df['option_b'][i]
            chosen = df['chosen'][i]
            rejected = df['rejected'][i]

            if option_a == chosen:
                true = "A"
            if option_b == chosen:
                true = "B"

            score = {'a':0, 'b':0}

            results_ab = []
            results_ba = []

            repeat = 0
            while has_duplicate_values(score):
                repeat += 1
                # Input AB and BA as a batch
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
                    # For AB input order
                    if j == 0: 
                        result = output.split("[/INST]")[1]
                        results_ab.append(result)
                        if len(result.split("Choice:")) == 2:
                            choice = result.split("Choice:")[1].strip()
                        else:
                            choice = "None"
                        if 'a' in choice.lower():
                            score['a'] += 1
                        else:
                            score['b'] += 1

                    # For BA input order
                    elif j == 1:
                        result = output.split("[/INST]")[1]
                        results_ba.append(result)
                        if len(result.split("Choice:")) == 2:
                            choice = result.split("Choice:")[1].strip()
                        else:
                            choice = "None"
                        if 'b' in choice.lower():
                            score['a'] += 1
                        else:
                            score['b'] += 1


                print(score)
                # If consistency is not achieved within 10 iterations
                if repeat >= 10:
                    break

            max_key = max(score, key=score.get)

            if max_key.lower() == true.lower():
                count += 1

            print(f">>> {count}/{i+1}")
            
            data[i] = {"item_no":str(item_no), "question":question, "answer":answer, "A":option_a, "B":option_b, "review_ab":results_ab, "review_ba":results_ba, "score":str(score), "choice":max_key.upper(), "true":true}

            with open(f'pairwise_ranking_result_{course}.json','w') as f:
                json.dump(data, f, indent=4)
