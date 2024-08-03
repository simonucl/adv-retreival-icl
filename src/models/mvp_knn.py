from random import triangular
from ..utils.model_utils import *
from ..utils.knn import HuggingFaceSentimentAnalysisPipelineWrapper
import os, pickle, copy
from torch_scatter import scatter_max, scatter_mean
import torch
from torch import nn
from .model_wrapper import ModelWrapper



class MVP_KNN(ModelWrapper):
    def __init__(self, args, base_model, mvp_model, tokenizer, data_collator, verbalizer = None, template=None):  
        '''
        args: args object from argparse
        model: huggingface model
        tokenizer: huggingface tokenizer
        data_collator: huggingface data collator
        verbalizer: dictoionary of verbalizer
        template: list of templates

        This is the MVP model
        '''
        label_words = []
        label_set = []
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        num_tokens = 1 if 'gpt' in args.model else 3 # bert tokenizes into cls, word, sep. we want the word to be a single token
        hidden_size = 1024 if 'large' in args.knn_model else 768
        base_model = base_model.to('cuda')
        self.knn_model = HuggingFaceSentimentAnalysisPipelineWrapper(tokenizer=tokenizer, model=base_model, model_dir=args.knn_model, 
                                                          ensemble_num=args.ensemble_num, sampled_num=args.sampled_num,
                                                            dataset=args.dataset, num_labels=args.num_labels,
                                                            batch_size=args.batch_size, tindex=args.tindex,
                                                            task=args.dataset, data_dir=args.data_dir,
                                                            hidden_size=hidden_size, train_epoch=args.train_epoch, shot=args.shot
                                                          )
        # import sys
        # sys.exit(1)

        # only keep those words that are tokenized into a single token
        for k,v in self.verbalizer.items():
            for word in v:
                if "roberta" in args.model:
                    word = " " + word
                if(len(self.tokenizer(word)["input_ids"]) == num_tokens):
                    label_set.append(k)
                    label_words.append(word)
                else:
                    print(word)
        self.label_set = torch.tensor(label_set)
        toks = self.tokenizer(label_words)["input_ids"]

        if 'gpt' not in args.model:
            new_toks = [t for t in toks if len(t) == num_tokens]
            self.label_word_ids = torch.tensor(new_toks)[:,1]
        else:
            new_toks = [t for t in toks]
            self.label_word_ids = torch.tensor(new_toks)[:,0]
        self.template_ids = []
        self.len_templates = []
        for prompt in template:
            used_prompt = prompt.replace("[MASK]", tokenizer.mask_token)
            if used_prompt.split(" ")[0] == "[SEP]":
                used_prompt = " ".join(used_prompt.split(" ")[1:])
            self.len_templates.append(1+len(tokenizer(used_prompt)["input_ids"][1:-1]))
        super(MVP_KNN, self).__init__(args, mvp_model, tokenizer, data_collator, verbalizer = verbalizer, template=template)

    def outs_to_logits(self, input_ids, outputs, raw_inputs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        outputs: output of the model
        raw_inputs: torch tensor of shape (batch_size, seq_len)

        returns logits of shape (batch_size, num_classes)
        '''
        logits = outputs.logits                             # (batch_size * num_templates, seq_len, vocab_size)
        batchid, indices = torch.where(input_ids == self.tokenizer.mask_token_id)
        if 'gpt' in self.args.model:
            # it predicts next word
            indices = indices -1

        mask_logits = logits[batchid, indices,:]         # (batch_size * num_templates, vocab_size)
        # print('Mask logits shape: ', mask_logits.shape)
        label_words_logits = mask_logits[:, self.label_word_ids]    # (batch_size * num_templates, num_candidates)
        # print('Label words logits shape: ', label_words_logits.shape)
        self.label_set = self.label_set.to(input_ids.device)
        if self.args.pool_label_words == "max":
            label_words_logits = scatter_max(label_words_logits, self.label_set)[0] # (batch_size * num_templates, num_classes)
        elif self.args.pool_label_words == "mean":
            label_words_logits = scatter_mean(label_words_logits, self.label_set)   # (batch_size * num_templates, num_classes)
        num_templates = 1 if (self.args.num_template == -2 and self.mode == "train") else len(self.template)
        template_mask = (torch.arange(label_words_logits.shape[0])/(num_templates)).to(torch.long)
        y = torch.stack([template_mask]*label_words_logits.shape[1],dim=1)
        y = y.to(input_ids.device)
        
        if self.args.pool_templates == "mean":
            label_words_logits = scatter_mean(label_words_logits, y, dim=0)   # (batch_size, num_classes)
        elif self.args.pool_templates == "max":
            label_words_logits = scatter_max(label_words_logits, y, dim=0)[0]  # (batch_size, num_classes)

        knn_logits = self.knn_model.outs_to_logits(raw_inputs)

        label_words_logits = torch.softmax(label_words_logits, dim=-1)
        knn_logits = torch.softmax(knn_logits, dim=-1)

        final_logits = self.args.beta * label_words_logits + (1-self.args.beta) * knn_logits

        # return final_logits
        print('Finished MVP_KNN outs_to_logits')
        return final_logits