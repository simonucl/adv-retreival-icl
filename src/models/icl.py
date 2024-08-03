from random import triangular
from ..utils.model_utils import *
from ..utils.knn import HuggingFaceSentimentAnalysisPipelineWrapper
import os, pickle, copy
from torch_scatter import scatter_max, scatter_mean
import torch
from torch import nn
from .model_wrapper import ModelWrapper
from ..utils.anchor import AnchorStore, subsamplebyshot, AnchorStores
from ..utils.dataset import *
from tqdm import tqdm
from time import time
from functools import lru_cache
from ..utils.dataset import format_template

SST2_LABELS2ID = {'0': 0, '1': 1}

class ICL(ModelWrapper):
    def __init__(self, args, model, tokenizer, data_collator, dataset, verbalizer = None, template=None):  
        '''
        args: args object from argparse
        model: huggingface model
        tokenizer: huggingface tokenizer
        data_collator: huggingface data collator
        verbalizer: dictoionary of verbalizer
        template: list of templates

        This is the MVP model
        '''
        super(ICL, self).__init__(args, model, tokenizer, data_collator, verbalizer = verbalizer, template=template)

        label_words = []
        label_set = []
        self.verbalizer = verbalizer
        self.inv_verbalizer = {v[0]:k for k,v in verbalizer.items()}

        self.tokenizer = tokenizer
        self.knn_T = args.knn_T
        self.knn_k = args.knn_k

        if 'gpt' in args.model:
            num_tokens = 1
        elif is_causal_model(args.model):
            num_tokens = 2
        else:
            num_tokens = 3

        # only keep those words that are tokenized into a single token
        for k,v in list(self.verbalizer.items()):
            for word in v:
                if (not is_causal_model(args.model)) or ('gpt' in args.model):
                    word = " " + word
                if(len(self.tokenizer(word)["input_ids"]) == num_tokens):
                    label_set.append(k)
                    label_words.append(word)
                else:
                    print(word)
                    assert len(self.tokenizer(word)["input_ids"]) == num_tokens, "Verbalizer word not tokenized into a single token"
        self.label_set = torch.tensor(label_set)
        toks = self.tokenizer(label_words)["input_ids"]

        # if args.dataset == "sst2":
        #     self.label2id = SST2_LABELS2ID
        # else:
        #     raise NotImplementedError
        
        if 'gpt' not in args.model:
            new_toks = [t for t in toks if len(t) == num_tokens]
            self.label_word_ids = torch.tensor(new_toks)[:,-1]
        else:
            new_toks = [t for t in toks]
            self.label_word_ids = torch.tensor(new_toks)[:,-1]
        self.template_ids = []
        self.len_templates = []

        anchor_data = dataset['train']
        self.anchor_data = anchor_data
        
        if args.model_type in ["icl", "icl_attack", "retrieval_icl_attack"]:
            examples_per_label = args.shot
        elif args.model_type in ["retrieval_icl"]:
            examples_per_label = 0
        else:
            examples_per_label = args.examples_per_label

        if (args.model_type in ["retrieval_icl"]) and (args.attack_name not in ["swap_labels", "icl_attack", "swap_orders", "irrelevant_sample"]):
            ralm_num = args.shot
            text_input_list = [(x['premise'], x['hypothesis']) if 'premise' in x.keys() else x['sentence'] for x in dataset[args.split]]

            icl_examples = self.indexEmbedder.subsamplebyretrieval(anchor_data, text_input_list, ralm_num, retrieve_method = args.retrieve_method, num_labels=len(verbalizer.keys()), save_path=args.ralm_save_path)
            anchor_subsample = []
        else:
            anchor_subsample, icl_examples = subsamplebyshot(anchor_data, args.seed, self.label_set, self.verbalizer, args.shot, examples_per_label)
        
            if args.save_icl_examples_path:
                concat_icl_examples = []
                for k, v in icl_examples.items():
                    concat_icl_examples += v
                with open(args.save_icl_examples_path, 'wb') as f:
                    pickle.dump(concat_icl_examples, f)
                import sys
                sys.exit(1)
        # print('Anchor subsample', anchor_data)

        # print('ICL examples', icl_examples)
        print('Length of anchor subsample', len(anchor_subsample))
        print('Length of icl examples', len(icl_examples))

        if self.args.model_type == "knn_cli":
            self.icl_examples = None
        else:
            self.icl_examples = icl_examples

        if not args.is_quantized:
            model = model.to('cuda')
            
        self.anchor_subsample = anchor_subsample

        if self.args.model_type in ["knn_icl"]:
            print("Loading anchor store")
            anchor_store = AnchorStore(
                                    K=(len(anchor_subsample))* (1 + int(self.args.adv_augment) + int(self.args.mask_augment)),
                                dim=model.config.hidden_size,
                                # dim=model.config.vocab_size,
                                knn=args.knn_k,
                                knn_T = args.knn_T,
                                n_class=args.num_labels
                                )
            self.anchor_store = anchor_store
 
            # print('Input sample example', anchor_subsample[0]['sentence'])

            for ins in tqdm(anchor_subsample, total=len(anchor_subsample)):
                labels = ins['label']
                # gen_logits = self.get_logits([ins['sentence']], labels)[0].detach().cpu()
                # self.anchor_store.enqueue(torch.softmax(gen_logits, dim=-1), torch.tensor(labels))
                input_ids = ins['sentence'] if 'sentence' in ins else (ins['premise'], ins['hypothesis'])
                hidden_states = self.get_logits([input_ids], labels, is_knn=True)[1].detach().cpu()
                self.anchor_store.enqueue(hidden_states, torch.tensor(labels))

                if args.adv_augment:
                    adv_gen_logits = self.get_logits([ins['sentence']], torch.tensor([labels]), adv=True).detach().cpu()
                    self.anchor_store.enqueue(torch.softmax(adv_gen_logits, dim=-1), torch.tensor(labels))
                if args.mask_augment:
                    mask_gen_logits = self.get_logits([ins['sentence']], torch.tensor([labels]), mask_augment=True).detach().cpu()
                    self.anchor_store.enqueue(torch.softmax(mask_gen_logits, dim=-1), torch.tensor(labels))
            print("Finished loading anchor store")
        elif self.args.model_type in ["knn_icl_attack"]:
            examples = []
            num_examples_per_label_map = [len(v) for k, v in icl_examples.items()]
            print('Num examples per label map', num_examples_per_label_map)
            # check if all instance in num_examples_per_label_map are equal
            if len(set(num_examples_per_label_map)) == 1:
                num_examples_per_label = num_examples_per_label_map[0]
                for idx in range(num_examples_per_label):
                    for label, example in icl_examples.items():
                        example = format_template(example[idx], template[0], self.args.dataset, label=label)
                        # example = example[idx]['sentence']
                        examples.append(example)
            else:
                for label, example in icl_examples.items():
                    for e in example:
                        examples.append(format_template(e, template[0], self.args.dataset, label=label))
            self.prompt = "\n\n".join(examples)

            anchor_store = AnchorStores(
                                    B=args.batch_size,
                                    K=(len(anchor_subsample))* (1 + int(self.args.adv_augment) + int(self.args.mask_augment)),
                                dim=model.config.hidden_size,
                                # dim=model.config.vocab_size,
                                knn=args.knn_k,
                                knn_T = args.knn_T,
                                n_class=args.num_labels
                                )
            self.anchor_store = anchor_store

    @lru_cache(maxsize=1000)
    def get_knn_logits(self, input_sent):

        sent = self.prompt + '\n\n' + input_sent
        # print("====================")
        # print(self.prompt)
        # print("====================")
        input_ids = self.tokenizer(sent, return_tensors='pt', padding=True, truncation=True)['input_ids']
        input_ids = input_ids.to('cuda')
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits[:, -1, :].detach().cpu()
        hidden_states = outputs.hidden_states[-1][:, -1, :].detach().cpu()

        return logits, hidden_states
    
    def get_logits(self, input_ids, labels=None, attention_mask=None, is_knn=False, adv=False, mask_augment=False, outputs=None, reduce_to_candidates=False):
        '''
        input_ids: torch tensor of shape (1, seq_len)
        attention_mask: torch tensor of shape (1, seq_len)
        '''

        if outputs is None:
            input_ids, attention_mask, input_ids_indices = self.get_updated_input_ids(input_ids, attention_mask, is_knn)
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        logits = outputs.logits                             # (B * num_templates, seq_len, vocab_size)        

        # if is_causal_model(self.args.model):
        #     # it predicts next word
        #     indices = indices -1

        # start_time = time()
        # last_nonpad_indices = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(dim=-1) - 1 # (B * num_templates)
        # last_nonpad_indices = last_nonpad_indices.to(logits.device)

        # mask_logits = logits[torch.arange(logits.shape[0]), last_nonpad_indices, :] # (B * num_templates, vocab_size)
        # mask_hidden_states = outputs.hidden_states[-1][torch.arange(logits.shape[0]), last_nonpad_indices, :] # (B * num_templates, hidden_size)

        mask_logits = logits[:, -1, :] # (B * num_templates, vocab_size)
        mask_hidden_states = outputs.hidden_states[-1][:, -1, :] # (B * num_templates, hidden_size)
        label_words_logits = mask_logits
        # end_time = time()

        # print('Time taken to get last nonpad indices', end_time - start_time)

        # mask_logits = logits[torch.arange(logits.shape[0]), last_nonpad_indices, :] # (B * num_templates, vocab_size)

        # mask_logits = logits[:, :, self.label_word_ids]    # (B * num_templates, seq_len, num_candidates
        
        
        if reduce_to_candidates:
            label_words_logits = mask_logits[:, self.label_word_ids]    # (B * num_templates, num_candidates)

            # This applicable when there's multiple label words
            # self.label_set = self.label_set.to(input_ids.device)
            # if self.args.pool_label_words == "max":
            #     label_words_logits = scatter_max(label_words_logits, self.label_set)[0] # (B * num_templates, num_classes)
            # elif self.args.pool_label_words == "mean":
            #     label_words_logits = scatter_mean(label_words_logits, self.label_set)   # (B * num_templates, num_classes)

        # All vocab or only the target words
        # num_templates = 1 if (self.args.num_template == -2 and self.mode == "train") else len(self.template)
        # template_mask = (torch.arange(label_words_logits.shape[0])/(num_templates)).to(torch.long)
        # y = torch.stack([template_mask]*label_words_logits.shape[1],dim=1)
        # y = y.to(input_ids.device)
        
        # if self.args.pool_templates == "mean":
        #     label_words_logits = scatter_mean(label_words_logits, y, dim=0)   # (B, vocab_size)
        # elif self.args.pool_templates == "max":
        #     label_words_logits = scatter_max(label_words_logits, y, dim=0)[0]  # (B, vocab_size)

        return label_words_logits, mask_hidden_states
    
    def outs_to_logits(self, input_ids, outputs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        outputs: output of the model
        raw_inputs: torch tensor of shape (batch_size, seq_len)

        returns logits of shape (batch_size, num_classes)
        '''

        query_logits, label_hidden_states = self.get_logits(input_ids, outputs=outputs) # (B, num_classes), (B, hidden_size)
        # print('Decoded output', self.tokenizer.batch_decode(torch.argmax(query_logits, dim=-1)))

        label_words_logits = query_logits[:, self.label_word_ids]    # (B, num_candidates)
        # label_words_logits = query_logits

        label_words_logits = torch.softmax(label_words_logits, dim=-1)
        if (self.args.model_type in ['knn_icl', 'knn_icl_attack']) and (self.args.beta > 0):
            
            # TODO think to improve the speed here to compute them together
            prob = self.anchor_store.knn_calibrate(label_hidden_states, dist_metric='l2')
            # check if the anchor store is a list of anchor store

            # prob = self.anchor_store.knn_calibrate(torch.softmax(query_logits, dim=-1), dist_metric='kl')

            label_words_logits = self.args.beta * prob + (1 - self.args.beta) * label_words_logits

        # print('ICL prob', label_words_logits)
        return label_words_logits