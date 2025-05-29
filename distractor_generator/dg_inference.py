"""
Inference code for the distractor generator.

The test set csv must include the following columns:
"item_no": test set item number, 
"question": question, 
"answer": correct answer, 
"options": human-authored distractors

The output JSON file is structured as follows:
"item_no": test set item number, 
"question": question, 
"answer": correct answer, 
"options": human-authored distractors, 
"types": type of distractor (Correct/Incorrect knowledge), 
"distractors": model-generated distractors
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
from torch.nn.utils.rnn import pad_sequence
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cleaning(text):
    text = text.lstrip('\n').rstrip('\n')
    text = text.replace('```python ',"").replace('```python\n',"").replace('```python',"")
    text = text.replace('```plain ',"").replace('```plain\n',"").replace('```plain',"")
    text = text.rstrip('```')
    text = text.lstrip('\n').rstrip('\n')

    return text

def extract_type_distractors(text):
    sections = text.split("###")[1:]  
    
    distractors = []
    for section in sections:
        if re.search(r"Type:", section):
            types = re.split(r"Type: ", section)[1]
            types = types.replace("\n", "")
        elif re.search(r"Distractor \d+:", section):
            distractor = re.split(r"Distractor \d+: ", section)[1]
            distractor = distractor[:-1] if distractor.endswith('\n') else distractor
            distractors.append(distractor)
    
    return types, distractors

prompt = """[INST] You are a teacher tasked with creating distractors (plausible wrong options) for a given Multiple Choice Question.
Generate distractors according to the guide below:
1) Distractor type:
- Analyze whether the question asks for a 'correct' or 'incorrect' option.
- If the question asks for a correct option, the distractor type should be "Incorrect knowledge"; if it asks for an incorrect option, the distractor type should be "Correct knowledge".
2) Distractors:
- The distractor should be well-formatted so that it fits naturally when presented together with the question and answer.
- If the distractor type is "Incorrect knowledge", the distractor must be an actually incorrect statement; if the distractor type is "Correct knowledge", the distractor must be an actually correct statement.

[Question]
{}

[Answer]
{}

Generate {} distractor(s) in the following format:
### Type: 
### Distractor n: [/INST]"""

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

course_list = ['python', "DB", "MLDL"]

for course in course_list:
    df = pd.read_csv(f'test_{course}.csv')  # Test set for each subject
    data = {}

    for i in range(len(df)):
        item_no = df['item_no'][i]
        question = df['question'][i]
        answer = df['answer'][i]
        answer = cleaning(answer)
        options = eval(df['options'][i])
        
        n = 3
        batch_inputs = [prompt.format(question, answer, n)]
        batch_encoded_inputs = tokenizer(batch_inputs, return_tensors='pt', padding=True, truncation=True).to(device)

        generate_kwargs = dict(
            input_ids=batch_encoded_inputs['input_ids'],
            attention_mask=batch_encoded_inputs['attention_mask'],
            do_sample=False,
            temperature=0,
            max_new_tokens=512,  # Reduce the number of tokens generated
            repetition_penalty=1.0
        )

        with torch.no_grad():
            batch_outputs = model.generate(**generate_kwargs)
        batch_decoded_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

        for j, output in enumerate(batch_decoded_outputs):
            #print(output)
            result = output.split("[/INST]")[1]
            types, distractors = extract_type_distractors(result)
            print(result)

        data[i] = {"item_no":str(item_no), "question":question, "answer":answer, "options":options, "types":types, "distractors":distractors}

        with open(f'distractor_inference_{course}.json','w') as f:
            json.dump(data, f, indent=4)
