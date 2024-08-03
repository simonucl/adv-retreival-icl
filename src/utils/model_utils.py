
import torch
import random
from .dataset import format_template

def insert_tokenized_template_front(tokenizer, model_type, input_id, template_id, len_templates):
    '''
    input_id: (batch, token_length)
    template_id: (token_length,)
    '''

    if "roberta" in model_type:
        template_id = tokenizer(" ")["input_ids"][1:-1] + template_id
    new_input_id = torch.zeros(min(tokenizer.model_max_length, input_id.shape[0]+max(len_templates)))
    new_input_id[0] = tokenizer.cls_token_id
    pad_indices = 1*(input_id==tokenizer.pad_token_id).nonzero()
    first_pad_index = 0
    if pad_indices.shape[0] > 0:
        first_pad_index = pad_indices[0].item()
    else:
        first_pad_index = input_id.shape[0]
    if(first_pad_index + len(template_id) < tokenizer.model_max_length):
        new_input_id[1:len(template_id)+1] = torch.tensor(template_id)
        new_input_id[len(template_id)+1:first_pad_index+len(template_id)] = input_id[1:first_pad_index]
        if first_pad_index+len(template_id) < new_input_id.shape[0]:
            new_input_id[first_pad_index+len(template_id):] = torch.tensor([tokenizer.pad_token_id]*new_input_id[first_pad_index+len(template_id):].shape[0])
    else:
        new_input_id[1:len(template_id)+1] = torch.tensor(template_id)
        new_input_id[len(template_id)+1:] = input_id[1:tokenizer.model_max_length - len(template_id)]
        new_input_id[-1] = tokenizer.sep_token_id
    new_attention_mask = 1*(new_input_id !=  tokenizer.pad_token_id)

    input_ids_indices = (len(template_id)+1, min(first_pad_index+len(template_id), tokenizer.model_max_length-1), -len(template_id)+1)
    return new_input_id, new_attention_mask, input_ids_indices

def insert_tokenized_template_back(tokenizer, model_type, input_id, template_id, len_templates):
    '''
    input_id: (batch, token_length)
    template_id: (token_length,)
    '''
    if "roberta" in model_type:
        template_id = tokenizer(" ")["input_ids"][1:-1] + template_id
    # create a new input d initialized with pad token id
    # new_input -> (token_length,)
    new_input_id = torch.ones(min(tokenizer.model_max_length, input_id.shape[0]+max(len_templates)))*tokenizer.pad_token_id
    # add cls token at the start
    if not is_causal_model(model_type):
        new_input_id[0] = tokenizer.cls_token_id 
    # find out all the pad_indices in the input_id
    pad_indices = 1*(input_id==tokenizer.pad_token_id).nonzero()
    first_pad_index = 0
    # find out the first pad index. If no pad then use the last sep token
    if pad_indices.shape[0] > 0:
        first_pad_index = pad_indices[0].item() - 1
    else:
        first_pad_index = input_id.shape[0] - 1
    if(first_pad_index + len(template_id) + 1 < tokenizer.model_max_length):
        new_input_id[:first_pad_index] = input_id[:first_pad_index]
        new_input_id[first_pad_index:first_pad_index+len(template_id)] = torch.tensor(template_id)
        if not is_causal_model(model_type):
            new_input_id[first_pad_index+len(template_id)] = tokenizer.sep_token_id
        if first_pad_index+len(template_id) < new_input_id.shape[0]:
            new_input_id[first_pad_index+len(template_id)+1:] = torch.tensor([tokenizer.pad_token_id]*new_input_id[first_pad_index+len(template_id)+1:].shape[0])
    else:
        # Truncate the input_id to model_max_length - len(template_id) - 1, if the input_id is too long
        new_input_id[:tokenizer.model_max_length-len(template_id)-1] = input_id[:tokenizer.model_max_length-len(template_id)-1]
        new_input_id[tokenizer.model_max_length-len(template_id)-1:tokenizer.model_max_length-1] = torch.tensor(template_id)
        if not is_causal_model(model_type):
            new_input_id[-1] = tokenizer.sep_token_id

    if is_causal_model(model_type):
        new_input_id[0] = tokenizer.bos_token_id
        
    input_ids_indices = (0, min(first_pad_index, tokenizer.model_max_length-len(template_id)-1), min(first_pad_index+len(template_id), tokenizer.model_max_length-1))

    new_attention_mask = 1*(new_input_id !=  tokenizer.pad_token_id)
    return new_input_id, new_attention_mask, input_ids_indices

def insert_tokenized_template_back_with_examples(tokenizer, model_type, input_id, template_id, len_templates, examples, max_len_examples=0):
    '''
    input_id: (batch, token_length)
    template_id: (token_length,)
    examples: {labels: input_ids}
    '''
    # sep_token_id = tokenizer(" ")["input_ids"][1]

    if "roberta" in model_type:
        template_id = tokenizer(" ")["input_ids"][1:-1] + template_id
    
    icl_total_length = 0
    demon_examples = []
    for v in examples:
        v_ids = tokenizer(v, padding=False, truncation=True)["input_ids"][1:-1] if not is_causal_model(model_type) else tokenizer(v, padding=False, truncation=True)["input_ids"][1:]
        # v_ids += [sep_token_id]

        demon_examples.append(torch.tensor(v_ids, dtype=torch.int64))

    # concat all the examples
    demon_examples = torch.concat(demon_examples, dim=0) # (demon_length,)
    # append a start of sentence token at the start
    # Add the sentence "Choose sentiment from Positive or Negative .\n" to the start of the demon examples
    instruction = "Choose sentiment from Positive or Negative .\n"
    instruction_ids = tokenizer(instruction, padding=False, truncation=True)["input_ids"][1:-1] if not is_causal_model(model_type) else tokenizer(instruction, padding=False, truncation=True)["input_ids"][1:]
    demon_examples = torch.cat([torch.tensor(instruction_ids), demon_examples], dim=0)
    if not is_causal_model(model_type):
        demon_examples = torch.cat([torch.tensor([tokenizer.cls_token_id]), demon_examples], dim=0)
    else:
        demon_examples = torch.cat([torch.tensor([tokenizer.bos_token_id]), demon_examples], dim=0)
        

    # print('Demon examples', tokenizer.decode(demon_examples))
    # print decoded demon examples
    demon_length = demon_examples.shape[0]

    # create a new input d initialized with pad token id
    # new_input -> (token_length,)
    new_input_id = torch.ones(min(tokenizer.model_max_length, max_len_examples+max(len_templates)+input_id.shape[0]))*tokenizer.pad_token_id
    # add cls token at the start
    pad_indices = 1*(input_id==tokenizer.pad_token_id).nonzero()
    first_pad_index = 0
    # find out the first pad index. If no pad then use the last sep token
    if pad_indices.shape[0] > 0:
        first_pad_index = pad_indices[0].item() - 1
    else:
        first_pad_index = input_id.shape[0] - 1

    # remove the bos and eos token from input_id
    input_id = input_id[1:-1] if not is_causal_model(model_type) else input_id[1:]
    template_id = template_id[1:]

    # TODO add the masking for input ids
    # added_length = len(added_ids) 
    # if self.mask_inference: 
    #     if random.random() > self.args.mask_prob:
    #         # mask_ratio = random.uniform(self.args.min_mask_ratio, 1.0) 
    #         mask_idx = random.sample(range(added_length), int(added_length * self.args.replace_ratio))
    #         choices_input_ids[choice_id] += [random.choice(range(len(self.tokenizer))) if _idx in mask_idx else added_ids[_idx] for _idx in range(added_length)]
    #         choices_attention_mask[choice_id] += [1] * added_length 
    #     else:
    #         mask_idx = random.sample(range(added_length), int(added_length * self.args.mask_ratio))
    #         choices_input_ids[choice_id] += added_ids
    #         choices_attention_mask[choice_id] += [0 if _idx in mask_idx else 1 for _idx in range(added_length)]


    if(first_pad_index + len(template_id) + 1 + demon_length < tokenizer.model_max_length):
        new_input_id[:demon_length] = demon_examples[:demon_length]
        new_input_id[demon_length:demon_length+first_pad_index] = input_id[:first_pad_index]

        input_id_indices = (demon_length, demon_length+first_pad_index, demon_length+first_pad_index+len(template_id))

        new_input_id[demon_length+first_pad_index:demon_length+first_pad_index+len(template_id)] = torch.tensor(template_id)
        if not is_causal_model(model_type):
            new_input_id[demon_length+first_pad_index+len(template_id)] = tokenizer.sep_token_id
        if demon_length+first_pad_index+len(template_id) < new_input_id.shape[0]:
            new_input_id[demon_length+first_pad_index+len(template_id)+1:] = torch.tensor([tokenizer.pad_token_id]*new_input_id[demon_length+first_pad_index+len(template_id)+1:].shape[0])
    else:
        # Truncate the input_id to model_max_length - len(template_id) - 1, if the input_id is too long
        exceed_len = demon_length + first_pad_index + len(template_id) + 1 - tokenizer.model_max_length

        if exceed_len > demon_length:
            new_input_id[:tokenizer.model_max_length-len(template_id)-1] = input_id[:tokenizer.model_max_length-len(template_id)-1]
            new_input_id[tokenizer.model_max_length-len(template_id)-1:tokenizer.model_max_length-1] = torch.tensor(template_id)

            input_id_indices = (0, tokenizer.model_max_length-len(template_id)-1, tokenizer.model_max_length-1)
        else:
            new_input_id[:demon_length-exceed_len] = demon_examples[exceed_len:demon_length]
            new_input_id[demon_length-exceed_len:tokenizer.model_max_length-len(template_id)-1] = input_id[:tokenizer.model_max_length-len(template_id)-1]
            new_input_id[tokenizer.model_max_length-len(template_id)-1:tokenizer.model_max_length-1] = torch.tensor(template_id)
            input_id_indices = (demon_length-exceed_len, tokenizer.model_max_length-len(template_id)-1, tokenizer.model_max_length-1)

        if not is_causal_model(model_type):
            new_input_id[-1] = tokenizer.sep_token_id

    new_attention_mask = 1*(new_input_id !=  tokenizer.pad_token_id)
    return new_input_id, new_attention_mask, input_id_indices

def insert_tokenized_prompts(tokenizer, model_type, text_input_list, templates, len_templates, use_all=True, icl_examples=None, len_examples=None):
    #input_ids, attention_mask = self.text_to_ids(text_input_list)]
    input_ids = text_input_list

    if icl_examples is not None:
        if len_examples is None:
            len_examples = []
        for template in templates:
            template_length = 0
            num_examples_per_label = len(list(icl_examples.values())[0])
            for idx in range(num_examples_per_label):
                for label, example in icl_examples.items():
                    # TODO Add support here when multiple examples are present
                    example = example[idx]['sentence']
                    example = example + template.replace("[MASK]", label)
                    template_length += len(tokenizer(example)["input_ids"]) - 1
            len_examples.append(template_length)
                    
    num_templates_used = len(templates) if use_all else 1
    max_icl_examples_len = max(len_examples) if icl_examples is not None else 0
    new_input_ids = torch.zeros(input_ids.shape[0]*num_templates_used, min(tokenizer.model_max_length, input_ids.shape[1]+max(len_templates)+max_icl_examples_len))
    new_attention_masks = torch.zeros(input_ids.shape[0]*num_templates_used, min(tokenizer.model_max_length, input_ids.shape[1]+max(len_templates)+max_icl_examples_len))
    new_input_id_indices = []
    for i in range(input_ids.shape[0]):
        if use_all:
            templates_new = templates
        else:
            templates_new = random.choices(templates, k=1)
        j = 0
        for template in templates_new:
            if icl_examples is not None:
                examples = []
                num_examples_per_label = len(list(icl_examples.values())[0])
                for idx in range(num_examples_per_label):
                    for label, example in icl_examples.items():
                        example = example[idx]['sentence']
                        example = example + template.replace("[MASK]", label)
                        examples.append(example)
            else:
                examples = None

            template = template.replace("[MASK]", tokenizer.mask_token)
            if template.split(" ")[0] == "[SEP]":
                template_ids = tokenizer(" ".join(template.split(" ")[1:]))["input_ids"][1:-1] if not is_causal_model(model_type) else tokenizer(" ".join(template.split(" ")[1:]))["input_ids"][1:]
                if examples is not None:
                    new_input_id, new_attention_mask, input_id_indices = insert_tokenized_template_back_with_examples(tokenizer, model_type, input_ids[i,:], template_ids, len_templates, examples, max_icl_examples_len)
                else:
                    new_input_id, new_attention_mask, input_id_indices = insert_tokenized_template_back(tokenizer, model_type, input_ids[i,:], template_ids, len_templates)
            else:
                template_ids = tokenizer(template)["input_ids"][1:-1] if not is_causal_model(model_type) else tokenizer(template)["input_ids"][1:]
                if examples is not None:
                    new_input_id, new_attention_mask, input_id_indices = insert_tokenized_template_back_with_examples(tokenizer, model_type, input_ids[i,:], template_ids, len_templates, examples, max_icl_examples_len)
                else:
                    new_input_id, new_attention_mask, input_id_indices = insert_tokenized_template_front(tokenizer, model_type, input_ids[i,:], template_ids, len_templates)
            
            new_input_ids[num_templates_used*i+j,:] =  new_input_id
            new_attention_masks[num_templates_used*i+j,:] =  new_attention_mask
            new_input_id_indices.append(input_id_indices)
            j=j+1
    return new_input_ids.long(), new_attention_masks.long(), new_input_id_indices

def insert_icl_prompts(model, tokenizer, model_type, text_input_list, templates, len_templates, use_all=True, icl_examples=None, len_examples=[0], model_name=""):
    '''
    '''
    # if icl_examples is not None:
    #     for template in templates:
    #         template = ("Classify the sentiment of {} and {}.\n".format(model.verbalizer[0][0], model.verbalizer[1][0])) + template
            
    #         num_examples_per_label = len(list(icl_examples.values())[0])
    #         for idx in range(num_examples_per_label):
    #             for label, example in icl_examples.items():
    #                 # TODO Add support here when multiple examples are present
    #                 example = example[idx]['sentence']
    #                 example = example + template.replace("[MASK]", label)
    #                 template_length += len(tokenizer(example)["input_ids"]) - 1
    #         len_examples.append(template_length)
                    
    num_templates_used = len(templates) if use_all else 1
    max_icl_examples_len = max(len_examples) if icl_examples is not None else 0

    input_length = []
    new_input_id_indices = []
    prompts = []
    # print('ICL Examples', icl_examples)
    for i in range(len(text_input_list)):
        if use_all:
            templates_new = templates
        else:
            templates_new = random.choices(templates, k=1)
        j = 0
        examples = []
        for template in templates_new:
            if icl_examples is not None:
                if (type(icl_examples) is list) and (type(icl_examples[0]) is list):
                    icl_example = icl_examples[i]
                    for e in icl_example:
                        example = format_template(e, template, model.args.dataset, verbalizer=model.verbalizer)
                        examples.append(example)
                else:
                    if type(icl_examples) is list:
                        icl_example = icl_examples[i]
                    else:
                        icl_example = icl_examples
                    
                    # assert if any of the label is ''
                    assert '' not in icl_example.keys(), "Empty label found in icl examples"
                    
                    num_examples_per_label_map = [len(v) for k, v in icl_example.items()]
                    # check if all instance in num_examples_per_label_map are equal
                    if len(set(num_examples_per_label_map)) == 1:
                        num_examples_per_label = num_examples_per_label_map[0]
                        for idx in range(num_examples_per_label):
                            for label, example in icl_example.items():
                                example = format_template(example[idx], template, model.args.dataset, label=label, verbalizer=model.verbalizer)
                                examples.append(example)
                    else:
                        for label, example in icl_example.items():
                            for e in example:
                                example = format_template(e, template, model.args.dataset, verbalizer=model.verbalizer)
                                examples.append(example)
                                # examples.append(template.format(e['sentence'], label))
            # print('Examples', examples)
            prompt = "\n\n".join(examples)

            # if model_type in ["knn_icl", "retrieval_icl", "retrieval_icl_attack"]:
            #     prompt_title = ""
            # else:
            #     prompt_title = "Classify the sentiment of {} and {}.\n\n".format(model.verbalizer[0][0], model.verbalizer[1][0])            
            prompt_title = ""
            if model.args.dataset in ["mnli", "cb"]:
                prompt_title = "Please identify whether the premise entails the hypothesis. The answer should be exact 'yes', 'no' or 'maybe'.\n\n"
            elif model.args.dataset in ["qqp"]:
                prompt_title = "Please identify whether the two questions are semantically equivalent. The answer should be exact 'yes' or 'no'.\n\n"
            else:
                prompt_title = ""
                
            input = text_input_list[i]
            if type(input) is tuple:
                premise, hypothesis = input
                inference_sample = "\n\n" + template.format(premise, hypothesis, "").strip()
            else:
                inference_sample = "\n\n" + template.format(input, "").strip()
            # if "gpt" in model.args.model:
            #     inference_sample += " "
            
            if "chat" in model_name:
                prompt = prompt_title + prompt
                prompt = "[INST] " + prompt + inference_sample + " [/INST]"
            elif "vicuna" in model_name:
                system_prompt = "A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input."
                prompt = system_prompt + " " + "USER: " + prompt_title + prompt + inference_sample + "\n" + "ASSISTANT: "
                # prompt = prompt_title + prompt
                # prompt = "<s> [INST] " + prompt + inference_sample + " [/INST] </s>"
            else:
                prompt = prompt_title + prompt + inference_sample
            
            prompts.append(prompt)

    # print('Prompts', prompts[0])
    # print('=============================================')
    # if model.args.model_type in ["icl_attack", "retrieval_icl"]:

    inputs = tokenizer.batch_encode_plus(prompts, padding=True, truncation=True, return_tensors="pt")

    # print('Decoded prompts', tokenizer.decode(inputs["input_ids"][0,:], skip_special_tokens=False))
    # print('Length of the decoded prompts', len(tokenizer.decode(inputs["input_ids"][0,:], skip_special_tokens=False)))
    
    new_input_ids = inputs["input_ids"]
    new_attention_masks = inputs["attention_mask"]

    return new_input_ids.long(), new_attention_masks.long(), new_input_id_indices

def craft_tokenized_prompts(tokenizer, model_type, input_ids, templates, len_templates, use_all=True, icl_examples=None, len_examples=[0]):
    '''
    input_ids: [[(examples, label), ... , inference], [(examples, label), ... , inference], ...]
    '''
    examples = []
    
    for i, template in enumerate(templates):
        for j in range(len(input_ids)):
            verbalized_example = []
            for example, label in input_ids[j][:-1]:
                verbalized_example.append(example + template.replace("[MASK]", label))
            verbalized_example.append(input_ids[j][-1] + template.replace("[MASK]", tokenizer.mask_token))
            examples.append(verbalized_example)
    
    print('Length of examples', len(examples))
    print('Examples', examples[0])
    num_templates_used = len(templates) if use_all else 1

    len_input_ids = []
    example_input_ids = []
    for example in examples:
        output = tokenizer(tokenizer.sep_token.join(example), padding=False, truncation=True, return_tensors="pt")
        example_input_ids.append(output["input_ids"])
        len_input_ids.append(output["input_ids"].shape[1])
        
    new_input_ids = torch.zeros(len(input_ids)*num_templates_used, min(tokenizer.model_max_length, max(len_input_ids)), dtype=torch.int64)
    new_attention_masks = torch.zeros(len(input_ids)*num_templates_used, min(tokenizer.model_max_length, max(len_input_ids)), dtype=torch.int64)
    new_input_id_indices = []

    for i in range(len(input_ids)):
        if use_all:
            templates_new = templates
        else:
            templates_new = random.choices(templates, k=1)
        for j in range(len(templates_new)):
            example = example_input_ids[num_templates_used*i+j]
            if example.shape[1] < tokenizer.model_max_length:
                new_input_ids[num_templates_used*i+j,:example.shape[1]] = example
                new_attention_masks[num_templates_used*i+j,:example.shape[1]] = torch.ones(example.shape[1])
            else:
                new_input_ids[num_templates_used*i+j,:tokenizer.model_max_length] = example[:,example.shape[1]-tokenizer.model_max_length:]
                new_attention_masks[num_templates_used*i+j,:tokenizer.model_max_length] = torch.ones(tokenizer.model_max_length)
            new_input_id_indices.append((0, example.shape[1]- len(templates_new[j].split(" ")), example.shape[1]))
    
    # print('New input ids', new_input_ids)
    # print('New attention mask', new_attention_masks)
    # print('New input id indices', new_input_id_indices)
    return new_input_ids.long(), new_attention_masks.long(), new_input_id_indices

def is_causal_model(model_type):
    return ("gpt" in model_type) or ("opt" in model_type) or ("Llama" in model_type) or ("Mistral" in model_type) or ("vicuna" in model_type) or ("gemma" in model_type)

