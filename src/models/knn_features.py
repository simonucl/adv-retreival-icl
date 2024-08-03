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

class KNNFeatures(ModelWrapper):
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
        super(KNNFeatures, self).__init__(args, model, tokenizer, data_collator, verbalizer = verbalizer, template=template)

        label_words = []
        label_set = []
        self.verbalizer = verbalizer
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

        anchor_data = dataset['train']

        if args.model_type in ["icl", "icl_attack"]:
            examples_per_label = args.shot
        elif args.model_type in ["retrieval_icl", "retrieval_icl_attack"]:
            examples_per_label = 0
        else:
            examples_per_label = args.examples_per_label


        anchor_subsample, icl_examples = subsamplebyshot(anchor_data, args.seed, self.label_set, self.verbalizer, args.shot, examples_per_label)
        
        print('Length of anchor subsample', len(anchor_subsample))
        print('Length of icl examples', len(icl_examples))

        if self.args.model_type == "knn_cli":
            self.icl_examples = None
        else:
            self.icl_examples = icl_examples

        model = model.to('cuda')
        self.anchor_subsample = anchor_subsample

        if self.args.model_type in ["knn_icl", "knn_icl_attack", "knn_features"]:
            print("Loading anchor store")
            anchor_store = AnchorStore(
                                    K=(len(anchor_subsample))* (1 + int(self.args.adv_augment) + int(self.args.mask_augment)),
                                dim=model.config.hidden_size,
                                knn=args.knn_k,
                                knn_T = args.knn_T,
                                n_class=args.num_labels
                                )
            self.anchor_store = anchor_store

            print('Input sample example', anchor_subsample[0]['sentence'])

            for ins in tqdm(anchor_subsample, total=len(anchor_subsample)):
                labels = ins['label']
                hidden_states = self.get_logits([ins['sentence']], labels)[1].detach().cpu()
                self.anchor_store.enqueue(hidden_states, torch.tensor(labels))

                # if args.adv_augment:
                #     adv_gen_logits = self.get_logits([ins['sentence']], torch.tensor([labels]), adv=True).detach().cpu()
                #     self.anchor_store.enqueue(torch.softmax(adv_gen_logits, dim=-1), torch.tensor(labels))
                # if args.mask_augment:
                #     mask_gen_logits = self.get_logits([ins['sentence']], torch.tensor([labels]), mask_augment=True).detach().cpu()
                #     self.anchor_store.enqueue(torch.softmax(mask_gen_logits, dim=-1), torch.tensor(labels))
            print("Finished loading anchor store")

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
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        logits = outputs.logits                             # (B * num_templates, seq_len, vocab_size)        
        hidden_state = outputs.hidden_states[-1]            # (B * num_templates, seq_len, hidden_size)
        # if is_causal_model(self.args.model):
        #     # it predicts next word
        #     indices = indices -1

        last_nonpad_indices = torch.ne(input_ids, self.tokenizer.pad_token_id).sum(dim=-1) - 1 # (B * num_templates)

        mask_logits = logits[torch.arange(logits.shape[0]), last_nonpad_indices, :] # (B * num_templates, vocab_size)
        mask_hidden_state = hidden_state[torch.arange(hidden_state.shape[0]), last_nonpad_indices, :] # (B * num_templates, hidden_size)
        label_words_logits = mask_logits
        # get the word output for each label_words_logits
        # pred = torch.argmax(logits, dim=-1) # (B * num_templates, seq_len)
        # pred = pred.cpu().numpy()
        # pred = self.tokenizer.batch_decode(pred)
        
        
        if reduce_to_candidates:
            label_words_logits = mask_logits[:, self.label_word_ids]    # (B * num_templates, num_candidates)

            # This applicable when there's multiple label words
            self.label_set = self.label_set.to(input_ids.device)
            if self.args.pool_label_words == "max":
                label_words_logits = scatter_max(label_words_logits, self.label_set)[0] # (B * num_templates, num_classes)
            elif self.args.pool_label_words == "mean":
                label_words_logits = scatter_mean(label_words_logits, self.label_set)   # (B * num_templates, num_classes)

        # All vocab or only the target words
        num_templates = 1 if (self.args.num_template == -2 and self.mode == "train") else len(self.template)
        template_mask = (torch.arange(label_words_logits.shape[0])/(num_templates)).to(torch.long)
        y = torch.stack([template_mask]*label_words_logits.shape[1],dim=1)
        y_features = torch.stack([template_mask]*mask_hidden_state.shape[1],dim=1)
        y = y.to(input_ids.device)
        y_features = y_features.to(input_ids.device)

        if self.args.pool_templates == "mean":
            label_words_logits = scatter_mean(label_words_logits, y, dim=0)   # (B, vocab_size)
            label_hidden_states = scatter_mean(mask_hidden_state, y_features, dim=0)   # (B, hidden_size)
        elif self.args.pool_templates == "max":
            label_words_logits = scatter_max(label_words_logits, y, dim=0)[0]  # (B, vocab_size)
            label_hidden_states = scatter_max(mask_hidden_state, y_features, dim=0)[0]

        return label_words_logits, label_hidden_states
    
    def outs_to_logits(self, input_ids, outputs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        outputs: output of the model
        raw_inputs: torch tensor of shape (batch_size, seq_len)

        returns logits of shape (batch_size, num_classes)
        '''

        query_logits, label_hidden_states = self.get_logits(input_ids, outputs=outputs)
        # print('Decoded output', self.tokenizer.batch_decode(torch.argmax(query_logits, dim=-1)))

        label_words_logits = query_logits[:, self.label_word_ids]    # (B, num_candidates)

        label_words_logits = torch.softmax(label_words_logits, dim=-1)
        if (self.args.model_type in ['knn_icl', 'knn_icl_attack']) and (self.args.beta > 0):
            query_logits = torch.softmax(query_logits, dim=-1)
            prob = self.anchor_store.knn_calibrate(query_logits)

            label_words_logits = self.args.beta * prob + (1 - self.args.beta) * label_words_logits
        elif (self.args.model_type in ['knn_features']):
            prob = self.anchor_store.knn_calibrate(label_hidden_states, dist_metric='l2')
            label_words_logits = self.args.beta * prob + (1 - self.args.beta) * label_words_logits
        # print('ICL prob', label_words_logits)
        return label_words_logits
