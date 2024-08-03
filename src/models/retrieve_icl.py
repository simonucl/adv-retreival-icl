from random import triangular
from ..utils.model_utils import *
from ..utils.knn import HuggingFaceSentimentAnalysisPipelineWrapper
import os, pickle, copy
from torch_scatter import scatter_max, scatter_mean
import torch
from torch import nn
from .model_wrapper import ModelWrapper
from ..utils.anchor import AnchorStore, subsamplebyshot
from ..utils.dataset import *
from tqdm import tqdm

SST2_LABELS2ID = {'0': 0, '1': 1}

class RETRIEVAL_ICL(ModelWrapper):
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
        super(RETRIEVAL_ICL, self).__init__(args, model, tokenizer, data_collator, verbalizer = verbalizer, template=template)

        label_words = []
        label_set = []
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.knn_T = args.knn_T
        self.knn_k = args.knn_k

        if 'gpt' in args.model:
            num_tokens = 1
        elif ('opt' in args.model) or ('Llama' in args.model) or ('Mistral' in args.model):
            num_tokens = 2
        else:
            num_tokens = 3

        # only keep those words that are tokenized into a single token
        for k,v in self.verbalizer.items():
            for word in v:
                if ('Llama' not in args.model) and ('Mistral' not in args.model):
                    word = " " + word
                if(len(self.tokenizer(word)["input_ids"]) == num_tokens):
                    label_set.append(k)
                    label_words.append(word)
                else:
                    print(word)
                    assert len(self.tokenizer(word)["input_ids"]) == num_tokens, "Verbalizer word not tokenized into a single token"
        self.label_set = torch.tensor(label_set)
        toks = self.tokenizer(label_words)["input_ids"]

        if args.dataset == "sst2":
            self.label2id = SST2_LABELS2ID
        else:
            raise NotImplementedError
        
        if 'gpt' not in args.model:
            new_toks = [t for t in toks if len(t) == num_tokens]
            self.label_word_ids = torch.tensor(new_toks)[:,1]
        else:
            new_toks = [t for t in toks]
            self.label_word_ids = torch.tensor(new_toks)[:,0]
        self.template_ids = []
        self.len_templates = []
        # for prompt in template:
        #     used_prompt = prompt.replace("[MASK]", tokenizer.mask_token)
        #     if used_prompt.split(" ")[0] == "[SEP]":
        #         used_prompt = " ".join(used_prompt.split(" ")[1:])
        #     self.len_templates.append(1+len(tokenizer(used_prompt)["input_ids"][1:-1]))
        if self.args.model_type in ["icl", "retrieval_icl"]:
            args.examples_per_label = 0

        anchor_data = dataset['train']
        anchor_subsample, icl_examples = subsamplebyshot(anchor_data, args.seed, self.label_set, self.verbalizer, args.shot, 0)
        
        if self.args.model_type == "knn_cli":
            self.icl_examples = None
        else:
            self.icl_examples = icl_examples

        model = model.to('cuda')

        self.anchor_subsample = anchor_subsample

    def get_logits(self, input_ids, labels=None, attention_mask=None, adv=False, mask_augment=False, outputs=None, reduce_to_candidates=False):
        '''
        input_ids: torch tensor of shape (1, seq_len)
        attention_mask: torch tensor of shape (1, seq_len)
        '''

        if outputs is None:
            input_ids, attention_mask, input_ids_indices = self.get_updated_input_ids(input_ids, attention_mask)
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits                             # (B * num_templates, seq_len, vocab_size)        

        # if is_causal_model(self.args.model):
        #     # it predicts next word
        #     indices = indices -1

        mask_logits = logits[:, -1,:]         # (B * num_templates, vocab_size)
        label_words_logits = mask_logits
        # get the word output for each label_words_logits

        if reduce_to_candidates:
            label_words_logits = mask_logits[:, self.label_word_ids]    # (B * num_templates, num_candidates)

            self.label_set = self.label_set.to(input_ids.device)
            if self.args.pool_label_words == "max":
                label_words_logits = scatter_max(label_words_logits, self.label_set)[0] # (B * num_templates, num_classes)
            elif self.args.pool_label_words == "mean":
                label_words_logits = scatter_mean(label_words_logits, self.label_set)   # (B * num_templates, num_classes)

        # All vocab or only the target words
        num_templates = 1 if (self.args.num_template == -2 and self.mode == "train") else len(self.template)
        template_mask = (torch.arange(label_words_logits.shape[0])/(num_templates)).to(torch.long)
        y = torch.stack([template_mask]*label_words_logits.shape[1],dim=1)
        y = y.to(input_ids.device)
        
        if self.args.pool_templates == "mean":
            label_words_logits = scatter_mean(label_words_logits, y, dim=0)   # (B, vocab_size)
        elif self.args.pool_templates == "max":
            label_words_logits = scatter_max(label_words_logits, y, dim=0)[0]  # (B, vocab_size)

        return label_words_logits # (1, vocab_size)
    
    def outs_to_logits(self, input_ids, outputs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        outputs: output of the model
        raw_inputs: torch tensor of shape (batch_size, seq_len)

        returns logits of shape (batch_size, num_classes)
        '''

        query_logits = self.get_logits(input_ids, outputs=outputs, reduce_to_candidates=True)

        query_logits = torch.softmax(query_logits, dim=-1)

        prob = query_logits

        return prob