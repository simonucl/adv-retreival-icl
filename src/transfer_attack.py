from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, FalconForCausalLM
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
# from tqdm.auto import tqdm  # for notebooks
import os
from functools import partial
import pickle as pkl
import json
import yaml

tqdm.pandas()

def get_demo_and_question(text, dataset="rte"):
    demons = text.split("<SPLIT>")
    demons = [demon.split(":")[1].strip('\n ').strip('[]') for demon in demons]

    if dataset == "rte":
        question = (demons[0], demons[1], "")
        icl_examples = []
        demons = demons[2:]
        for i in range(len(demons) // 3): # Limited to NLI results for now
            icl_examples.append((demons[i * 3], demons[i * 3 + 1], demons[i * 3 + 2]))
    elif dataset == "trec":
        question = (demons[0], "")
        icl_examples = []
        demons = demons[1:]
        for i in range(len(demons) // 2):
            icl_examples.append((demons[i * 2], demons[i * 2 + 1]))
    elif dataset == "sst2":
        question = (demons[0], "")
        icl_examples = []
        demons = demons[1:]
        for i in range(len(demons) // 2):
            icl_examples.append((demons[i * 2], demons[i * 2 + 1]))
            
    return question, icl_examples
    

def get_prompt(row, text_col='original_text', verbalizer={0: "true", 1: "false"}, icl_examples_col='icl_examples', template=None, dataset="rte"):
    question, icl_examples = get_demo_and_question(row[text_col], dataset=dataset)
    if len(icl_examples) == 0:
        icl_examples = row[icl_examples_col]

    # if template is None:
    #     template = "{}\n The question is: {}. True or False?\nThe Answer is: {}"
    # verbalizer = {0: "true", 1: "false"}

    demos = []
    for demo in icl_examples:
        if dataset == "rte":
            if isinstance(demo, tuple):
                demos.append(template.format(demo[0], demo[1], demo[2]))
            elif isinstance(demo, dict):
                demos.append(template.format(demo['premise'], demo['hypothesis'], verbalizer[demo['label']]))
        elif dataset == "trec":
            if isinstance(demo, tuple):
                demos.append(template.format(demo[0], demo[1]))
            elif isinstance(demo, dict):
                demos.append(template.format(demo['example'], verbalizer[demo['label']]))
        elif dataset == "sst2":
            if isinstance(demo, tuple):
                demos.append(template.format(demo[0], demo[1]))
            elif isinstance(demo, dict):
                demos.append(template.format(demo['example'], verbalizer[demo['label']]))

    if dataset == "rte":
        q = template.format(question[0], question[1], "").strip()
    elif dataset in ["trec", "sst2"]:
        q = template.format(question[0], "").strip()

    prompt = "\n\n".join(demos) + "\n\n" + q

    return prompt

# Making sure the perturbed text is not changing the question and the demonstrations
def compare_non_modifable(row):
    original = row['original_text']
    modified = row['perturbed_text']
    ori_q, ori_icl_examples = get_demo_and_question(original)
    mod_q, mod_icl_examples = get_demo_and_question(modified)

    return (all([(e[0] == ae[0]) and (e[1] == ae[1]) for e, ae in zip(ori_icl_examples, mod_icl_examples)])) and (ori_q == mod_q)

def compute_distributions(question, icl_examples, tokenizer, model, verbalizer, template=None, prompt=None, dataset="rte"):
    model.eval()
    label_id = [tokenizer.encode(v)[1] for k, v in verbalizer.items()]

    # if isinstance(model, FalconForCausalLM):
    #     label_id = [tokenizer.encode(' ' + verbalizer[0])[0], tokenizer.encode(' ' + verbalizer[1])[0]]
    # else:
    #     label_id = [tokenizer.encode(verbalizer[0])[1], tokenizer.encode(verbalizer[1])[1]]
        
    if prompt is None:
        if template is None:
            template = "{}\n The question is: {}. True or False?\nThe Answer is: {}"
        # template = "{}\n The question is: {}. True or False?\nThe Answer is: {}"

        demos = []
        for demo in icl_examples:
            if len(demo) == 2:
                demos.append(template.format(demo[0], demo[1]))
            else:
                demos.append(template.format(demo[0], demo[1], demo[2]))
        if len(question) == 1:
            q = template.format(question[0], "").strip()
        else:
            q = template.format(question[0], question[1], "").strip()

        prompt = "\n\n".join(demos) + "\n\n" + q

    # print(prompt)
    tokenized = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=1536).to(model.device)
    # print(f'Tokenied length: {len(tokenized["input_ids"][0])}')

    logits = model(**tokenized).logits
    output = logits[:, -1, :].detach().cpu()

    # print('decoded:' + tokenizer.decode(output.argmax(dim=-1).item()))
    output_label = output[:, label_id].softmax(dim=-1)
    return output_label.argmax(dim=-1).item()

def compute_the_attacked_answer(row, tokenizer, model, verbalizer, dataset):
    if 'perturbed_prompt' in row:
        prompt = row['perturbed_prompt']
        return compute_distributions(None, None, tokenizer=tokenizer, model=model, prompt=prompt, verbalizer=verbalizer)
    else:
        original = row['original_text']
        modified = row['perturbed_text']
        # ori_q, ori_icl_examples = get_demo_and_question(original)
        mod_q, mod_icl_examples = get_demo_and_question(modified, dataset=dataset)

        return compute_distributions(mod_q, mod_icl_examples, tokenizer=tokenizer, model=model, verbalizer=verbalizer)

def compute_original_answer(row, tokenizer, model, verbalizer, dataset):
    if 'original_prompt' in row:
        prompt = row['original_prompt']
        return compute_distributions(None, None, tokenizer=tokenizer, model=model, prompt=prompt, verbalizer=verbalizer)
    else:
        original = row['original_text']
        modified = row['perturbed_text']
        ori_q, ori_icl_examples = get_demo_and_question(original)
        # mod_q, mod_icl_examples = get_demo_and_question(modified)

        return compute_distributions(ori_q, ori_icl_examples, tokenizer=tokenizer, model=model, verbalizer=verbalizer)

def random_flip(icl_examples, percentage, verbalizer={0: "true", 1: "false"}):
    np.random.seed(1)
    idx = np.random.choice(len(icl_examples), int(len(icl_examples) * percentage), replace=False)
    for i in idx:
        if len(icl_examples[i]) == 2:
            icl_examples[i] = (icl_examples[i][0], icl_examples[i][1], verbalizer[0] if icl_examples[i][1] == verbalizer[1] else verbalizer[1])
        else:
            icl_examples[i] = (icl_examples[i][0], icl_examples[i][1], 'false' if icl_examples[i][2] == 'true' else 'true')

    return icl_examples

def fully_flip(row, tokenizer, model, verbalizer, label='false', dataset='rte', template=None):
    original = row['original_text']
    ori_q, ori_icl_examples = get_demo_and_question(original, dataset=dataset)
    for i in range(len(ori_icl_examples)):
        if dataset == 'rte':
            ori_icl_examples[i] = (ori_icl_examples[i][0], ori_icl_examples[i][1], label)
        elif dataset in ['trec', 'sst2']:
            ori_icl_examples[i] = (ori_icl_examples[i][0], label)

    return compute_distributions(ori_q, ori_icl_examples, tokenizer=tokenizer, model=model, template=template, verbalizer=verbalizer, dataset=dataset)

def compute_random_flip_original_answer(row, tokenizer, model, template, dataset, verbalizer):
    original = row['original_text']
    ori_q, ori_icl_examples = get_demo_and_question(original, dataset=dataset)
    ori_icl_examples = random_flip(ori_icl_examples, 1.0, verbalizer=verbalizer)
    # mod_q, mod_icl_examples = get_demo_and_question(modified)

    return compute_distributions(ori_q, ori_icl_examples, template=template, tokenizer=tokenizer, model=model, dataset=dataset, verbalizer=verbalizer)

def compute_swap_labels_details(df, tokenizer, model, out_path, model_name, metrics, template, dataset, verbalizer):
    df['random_flip_original_answer'] = df.progress_apply(lambda x : compute_random_flip_original_answer(x, tokenizer=tokenizer, model=model, template=template, dataset=dataset, verbalizer=verbalizer), axis=1)
    df['full_flip_true_original_answer'] = df.progress_apply(lambda row: fully_flip(row, tokenizer=tokenizer, model=model, label=verbalizer[1], dataset=dataset, verbalizer=verbalizer, template=template), axis=1)
    df['full_flip_false_original_answer'] = df.progress_apply(lambda row: fully_flip(row, tokenizer=tokenizer, model=model, label=verbalizer[0], dataset=dataset, verbalizer=verbalizer, template=template), axis=1)

    df['random_flip_correct'] = df['random_flip_original_answer'] == df['ground_truth_output']
    df['full_flip_true_correct'] = df['full_flip_true_original_answer'] == df['ground_truth_output']
    df['full_flip_false_correct'] = df['full_flip_false_original_answer'] == df['ground_truth_output']

    print('\nRandom Flip Accuracy')
    print(round(df['random_flip_correct'].value_counts()[True] / df['random_flip_correct'].value_counts().sum(), 4))

    print('\nAll True Accuracy')
    print(round(df['full_flip_true_correct'].value_counts()[True] / df['full_flip_true_correct'].value_counts().sum(), 4))
    print('\nAll False Accuracy')
    print(round(df['full_flip_false_correct'].value_counts()[True] / df['full_flip_false_correct'].value_counts().sum(), 4))
    
    metrics['random_flip_acc'] = df['random_flip_correct'].value_counts()[True] / df['random_flip_correct'].value_counts().sum()
    metrics['full_flip_true_acc'] = df['full_flip_true_correct'].value_counts()[True] / df['full_flip_true_correct'].value_counts().sum()
    metrics['full_flip_false_acc'] = df['full_flip_false_correct'].value_counts()[True] / df['full_flip_false_correct'].value_counts().sum()

    # return metrics
    from collections import Counter

    def compute_label_icl_example_dist(row):
        # if row['result_type'] == 'Skipped':
        #     return {}
        
        modified = row['perturbed_text']
        mod_q, mod_icl_examples = get_demo_and_question(modified)

        return dict(Counter([e[2] for e in mod_icl_examples]))

    df['attack_demonstrations_dist'] = df.apply(compute_label_icl_example_dist, axis=1)
    df['perturbed_examples'] = df.progress_apply(lambda x : get_demo_and_question(x['perturbed_text'])[1], axis=1)

    successful_attack = df[df['result_type'] == 'Successful']

    def get_the_label_dist(row):
        demo_dist = row['attack_demonstrations_dist']
        # print(demo_dist)
        if demo_dist == {}:
            return {}
        correct_answer = 'false' if row['ground_truth_output'] == 0 else 'true'

        return {correct_answer: demo_dist[correct_answer]}

    successful_attack['correct_label_dist'] = successful_attack.apply(get_the_label_dist, axis=1)

    mapping = {0: 'false', 1: 'true'}

    # measure the correct_label_dist based on ground_truth_output and plot them on a line chart
    # successful_attack = successful_attack['correct_label_dist'].apply(lambda x: {mapping[k]: v for k, v in x.items()})


    buckets = {'true': [], 'false': []}
    for i, row in successful_attack.iterrows():
        for k, v in row['correct_label_dist'].items():
            buckets[k].append(v-args.shot)

    # draw them on a 2d bar chart

    # final_bucket = buckets['true'] + [-1 * v for v in buckets['false']]

    # plot the histogram with larger than zero as green and smaller than zero as red
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_histogram(buckets):
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.bar([ x +0.25 for x in Counter(buckets['true']).keys()], [x / len(buckets['true']) for x in Counter(buckets['true']).values()], color='green', alpha=0.5, label='True', width=0.5)
        plt.bar([ -1 * (x + 0.25) for x in Counter(buckets['false']).keys()], [x / len(buckets['false']) for x in Counter(buckets['false']).values()], color='red', alpha=0.5, label='False', width=0.5)
        # plt.hist([-1 * b for b in buckets['false']], bins=25, color='red', alpha=0.5, label='False')
        plt.title("Histogram of Successful Attacks")
        plt.xlabel("Number of Positive Demonstrations")
        # make x axis as discrete values
        plt.xticks(np.arange(-8, 9, 1))
        plt.ylabel("Count")
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(out_path, 'successful_attack_histogram.png'))

    plot_histogram(buckets)
    df.to_csv(os.path.join(out_path, f'{model_name}_attack_results.csv'), index=False)

def get_verbalizer_and_template(dataset):
    verbalizer = yaml.safe_load(open(f'./configs/verbalizer_{dataset}.yaml', 'r'))
    template = list(yaml.safe_load(open(f'./configs/templates_{dataset}.yaml', 'r')).values())[0]
    verbalizer = {
        int(k): v[0] if isinstance(v, list) else v for k, v in verbalizer.items()
    }
    return verbalizer, template

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(args):

    print('Loading data')
    df = pd.read_csv(args.csv_path)
    out_path = '/'.join(args.csv_path.split('/')[:-1])
    is_fix_dist = 'fix_dist' in args.csv_path

    model_name = args.model.split('/')[-1]

    if args.demonstration_path:
        icl_examples = pkl.load(open(args.demonstration_path, 'rb'))
        # check if icl_examples is a list of list of dictionaries
        if isinstance(icl_examples[0], list):
            df['icl_examples'] = icl_examples
        else:
            icl_examples = [icl_examples] * len(df)
            df['icl_examples'] = icl_examples
        if args.add_icl_examples_only:
            out_path_split = out_path.split('/')
            output_path = '/'.join(out_path_split[:4]) + '/analysis/' + '/'.join(out_path_split[4:])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df.to_csv(os.path.join(output_path, f'{model_name}_analysis.csv'), index=False)
            return
        
    verbalizer, template = get_verbalizer_and_template(args.dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('Loading model')
    if args.precision == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(args.model, use_flash_attention_2=True, torch_dtype=torch.bfloat16, device_map='auto')
        model = model.to('cuda')
    elif args.precision == 'int8':
        model = AutoModelForCausalLM.from_pretrained(args.model, use_flash_attention_2=True, load_in_8bit=True, device_map='auto')
    elif args.precision == 'int4':
        model = AutoModelForCausalLM.from_pretrained(args.model, use_flash_attention_2=True, load_in_4bit=True, device_map='auto')

    model.eval()
    # tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    
    df['original_prompt'] = df.progress_apply(partial(get_prompt, text_col='original_text', template=template, dataset=args.dataset, verbalizer=verbalizer), axis=1)
    df['perturbed_prompt'] = df.progress_apply(partial(get_prompt, text_col='perturbed_text', template=template, dataset=args.dataset, verbalizer=verbalizer), axis=1)

    # df['non_modifiable'] = df.progress_apply(compare_non_modifable, axis=1)
    df['attacked_answer'] = df.progress_apply(lambda x : compute_the_attacked_answer(x, tokenizer=tokenizer, model=model, verbalizer=verbalizer, dataset=args.dataset), axis=1)
    df['original_answer'] = df.progress_apply(lambda x : compute_original_answer(x, tokenizer=tokenizer, model=model, verbalizer=verbalizer, dataset=args.dataset), axis=1)

    df['correct'] = df['original_answer'] == df['ground_truth_output']
    df['attack_correct'] = df['attacked_answer'] == df['ground_truth_output']

    metrics = {}

    clean_acc = df['correct'].value_counts()[True] / df['correct'].value_counts().sum()
    attack_acc = df['attack_correct'].value_counts()[True] / df['attack_correct'].value_counts().sum()
    asr = (clean_acc - attack_acc) / clean_acc

    print('Original Accuracy')
    print(round(clean_acc, 4))
    print('\nAttack Accuracy')
    print(round(attack_acc, 4))
    print('\nASR')
    print(round(asr, 4))
    
    metrics['clean_acc'] = clean_acc
    metrics['attack_acc'] = attack_acc
    metrics['asr'] = asr

    if is_fix_dist:
        model_name = model_name + '_fix_dist'
        
    df.to_csv(os.path.join(out_path, f'{model_name}_attack_results.csv'), index=False)
    # save the metrics as json

    if args.attack == 'swap_labels' or args.attack == 'swap_labels_fix_dist':
        metrics = compute_swap_labels_details(df, tokenizer, model, out_path, model_name, metrics, template, args.dataset, verbalizer)

    with open(os.path.join(out_path, f'{model_name}_attack_metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='lmsys/vicuna-7b-v1.5')
    args.add_argument('--csv_path', type=str, required=True)
    args.add_argument('--precision', type=str, default='bf16')
    args.add_argument('--demonstration_path', type=str, default=None)
    args.add_argument('--attack', type=str, default='swap_labels')
    args.add_argument('--shot', type=int, default=8)
    args.add_argument('--dataset', type=str, default='rte')
    args.add_argument('--add_icl_examples_only', action='store_true')

    args = args.parse_args()

    main(args)