from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, use_flash_attention_2=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_flash_attention_2=True, torch_dtype=torch.bfloat16)
model = model.to('cuda')
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
tokenizer.pad_token = tokenizer.eos_token

import pandas as pd
import numpy as np
from tqdm.auto import tqdm  # for notebooks

tqdm.pandas()

input_file = "../checkpoints/rte/meta-llama/Llama-2-7b-hf/swap_labels/icl_attack-seed-1-shot-8_quantized_bound/swap_labels_log.csv"
df = pd.read_csv(input_file)

def get_demo_and_question(text):
    demons = text.split("<SPLIT>")
    demons = [demon.split(":")[1].strip('\n ').strip('[]') for demon in demons]

    question = (demons[0], demons[1], "")
    icl_examples = []
    demons = demons[2:]
    for i in range(len(demons) // 3):
        icl_examples.append((demons[i * 3], demons[i * 3 + 1], demons[i * 3 + 2]))
    return question, icl_examples

def get_prompt(text):
    question, icl_examples = get_demo_and_question(text)
    template = "{}\n The question is: {}. True or False?\nThe Answer is: {}"
    verbalizer = {0: "true", 1: "false"}

    demos = []
    for demo in icl_examples:
        demos.append(template.format(demo[0], demo[1], demo[2]))
    q = template.format(question[0], question[1], "").strip()

    prompt = "\n\n".join(demos) + "\n\n" + q

    return prompt

def compare_non_modifable(row):
    original = row['original_text']
    modified = row['perturbed_text']
    ori_q, ori_icl_examples = get_demo_and_question(original)
    mod_q, mod_icl_examples = get_demo_and_question(modified)

    return (all([(e[0] == ae[0]) and (e[1] == ae[1]) for e, ae in zip(ori_icl_examples, mod_icl_examples)])) and (ori_q == mod_q)

def compute_distributions(question, icl_examples):
    template = "{}\n The question is: {}. True or False?\nThe Answer is: {}"
    verbalizer = {0: "true", 1: "false"}
    label_id = [tokenizer.encode(verbalizer[0])[1], tokenizer.encode(verbalizer[1])[1]]

    demos = []
    for demo in icl_examples:
        demos.append(template.format(demo[0], demo[1], demo[2]))
    q = template.format(question[0], question[1], "").strip()

    prompt = "\n\n".join(demos) + "\n\n" + q

    # print(prompt)
    tokenized = tokenizer(prompt, return_tensors="pt", padding=True).to('cuda')
    logits = model(**tokenized).logits
    output = logits[:, -1, :].detach().cpu()

    output_label = output[:, label_id].softmax(dim=-1)
    return output_label.argmax(dim=-1).item()

def compute_the_attacked_answer(row):
    if row['result_type'] == 'Skipped':
        return -1
    
    original = row['original_text']
    modified = row['perturbed_text']
    # ori_q, ori_icl_examples = get_demo_and_question(original)
    mod_q, mod_icl_examples = get_demo_and_question(modified)

    return compute_distributions(mod_q, mod_icl_examples)

def compute_original_answer(row):
    if row['result_type'] == 'Skipped':
        return -1
    
    original = row['original_text']
    modified = row['perturbed_text']
    ori_q, ori_icl_examples = get_demo_and_question(original)
    # mod_q, mod_icl_examples = get_demo_and_question(modified)

    return compute_distributions(ori_q, ori_icl_examples)

df['non_modifiable'] = df.progress_apply(compare_non_modifable, axis=1)
df['attacked_answer'] = df.progress_apply(compute_the_attacked_answer, axis=1)
df['original_answer'] = df.progress_apply(compute_original_answer, axis=1)

def random_flip(icl_examples, percentage):
    np.random.seed(1)
    idx = np.random.choice(len(icl_examples), int(len(icl_examples) * percentage), replace=False)
    for i in idx:
        icl_examples[i] = (icl_examples[i][0], icl_examples[i][1], 'false' if icl_examples[i][2] == 'true' else 'true')

    return icl_examples

def fully_flip(row, label='false'):
    original = row['original_text']
    ori_q, ori_icl_examples = get_demo_and_question(original)
    for i in range(len(ori_icl_examples)):
        ori_icl_examples[i] = (ori_icl_examples[i][0], ori_icl_examples[i][1], label)

    return compute_distributions(ori_q, ori_icl_examples)

def compute_random_flip_original_answer(row):
    if row['result_type'] == 'Skipped':
        return -1
    
    original = row['original_text']
    ori_q, ori_icl_examples = get_demo_and_question(original)
    ori_icl_examples = random_flip(ori_icl_examples, 0.5)
    # mod_q, mod_icl_examples = get_demo_and_question(modified)

    return compute_distributions(ori_q, ori_icl_examples)

df['random_flip_original_answer'] = df.progress_apply(compute_random_flip_original_answer, axis=1)
df['full_flip_true_original_answer'] = df.progress_apply(lambda row: fully_flip(row, 'true'), axis=1)
df['full_flip_true_original_answer'] = df.progress_apply(lambda row: fully_flip(row, 'false'), axis=1)

df['correct'] = df['original_answer'] == df['ground_truth_output']
df['attack_correct'] = df['attacked_answer'] == df['ground_truth_output']
df['random_flip_correct'] = df['random_flip_original_answer'] == df['ground_truth_output']

print('Original Accuracy')
print(df['correct'].value_counts())
print('\nAttack Accuracy')
print(df['attack_correct'].value_counts())
print('\nRandom Flip Accuracy')
print(df['random_flip_correct'].value_counts())