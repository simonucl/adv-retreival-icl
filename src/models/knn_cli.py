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
from ..utils.augmentations import *

SST2_LABELS2ID = {'0': 0, '1': 1}

class KNN_CLI(ModelWrapper):
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
        super(KNN_CLI, self).__init__(args, model, tokenizer, data_collator, verbalizer = verbalizer, template=template)

        label_words = []
        label_set = []
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        self.knn_T = args.knn_T
        self.knn_k = args.knn_k

        if 'gpt' in args.model:
            num_tokens = 1
        elif ('opt' in args.model) or ('Llama' in args.model):
            num_tokens = 2
        else:
            num_tokens = 3

        # only keep those words that are tokenized into a single token
        for k,v in self.verbalizer.items():
            for word in v:
                if not is_causal_model(args.model):
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
        for prompt in template:
            used_prompt = prompt.replace("[MASK]", tokenizer.mask_token)
            if used_prompt.split(" ")[0] == "[SEP]":
                used_prompt = " ".join(used_prompt.split(" ")[1:])
            self.len_templates.append(1+len(tokenizer(used_prompt)["input_ids"][1:-1]))

        anchor_data = dataset['train']
        anchor_subsample, icl_examples = subsamplebyshot(anchor_data, args.seed, self.label_set, self.verbalizer, args.shot, args.examples_per_label)
        
        print('ICL examples', icl_examples)

        if self.args.model_type == "knn_cli":
            self.icl_examples = None
        else:
            self.icl_examples = icl_examples

        model = model.to('cuda')

        print("Loading anchor store")
        anchor_store = AnchorStore(
                                K=len(anchor_subsample)* (1 + int(self.args.adv_augment) + int(self.args.mask_augment)),
                               dim=model.config.vocab_size,
                               knn=args.knn_k,
                               knn_T = args.knn_T,
                               n_class=args.num_labels
                               )
        self.anchor_store = anchor_store

        print('Input sample example', anchor_subsample[0]['sentence'])

        for ins in tqdm(anchor_subsample, total=len(anchor_subsample)):
            labels = ins['label']
            gen_logits = self.get_logits([ins['sentence']], labels).detach().cpu()
            self.anchor_store.enqueue(torch.softmax(gen_logits, dim=-1), torch.tensor(labels))

            if args.adv_augment:
                adv_gen_logits = self.get_logits([ins['sentence']], torch.tensor([labels]), adv=True).detach().cpu()
                self.anchor_store.enqueue(torch.softmax(adv_gen_logits, dim=-1), torch.tensor(labels))
            if args.mask_augment:
                mask_gen_logits = self.get_logits([ins['sentence']], torch.tensor([labels]), mask_augment=True).detach().cpu()
                self.anchor_store.enqueue(torch.softmax(mask_gen_logits, dim=-1), torch.tensor(labels))
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

            if adv:
                assert labels is not None
                new_input_ids = torch.zeros(len(input_ids_indices), max([x[2]-x[0] for x in input_ids_indices]), dtype=torch.int64)
                new_input_ids = new_input_ids.to('cuda')
                new_attention_mask = torch.zeros(len(input_ids_indices), max([x[2]-x[0] for x in input_ids_indices]), dtype=torch.int64)
                new_attention_mask = new_attention_mask.to('cuda')
                # fill new_input_ids with self.tokenizer.pad_token_id
                new_input_ids = new_input_ids.fill_(self.tokenizer.pad_token_id)
                
                for i, (start, end, template) in enumerate(input_ids_indices):
                    '''
                    start and end are the indices of the input_ids before the template is inserted
                    template is pointing to the end of the template
                    '''
                    new_input_ids[i, :template-start] = input_ids[i, start:template]
                    new_attention_mask[i, :template-start] = attention_mask[i, start:template]

                embedding_outs = pgd_attack(self, new_input_ids, new_attention_mask, labels, self.args, norm = self.args.norm)
                word_embedding_layer = self.model.get_input_embeddings()
                new_embedding_outs = word_embedding_layer(input_ids)
                
                for i, (start, end, template) in enumerate(input_ids_indices):
                    new_embedding_outs[i, start:template] = embedding_outs[i, :template-start]
                new_embedding_outs = new_embedding_outs.to('cuda')
                with torch.no_grad():
                    outputs = self.model(inputs_embeds=new_embedding_outs, attention_mask=attention_mask)
            elif mask_augment:
                for choice_id in range(len(input_ids)):
                    if random.random() > self.args.mask_prob:
                        # mask_ratio = random.uniform(self.args.min_mask_ratio, 1.0) 
                        start, end, _ = input_ids_indices[choice_id]
                        added_length = end - start
                        mask_idx = random.sample(range(added_length), int(added_length * self.args.replace_ratio))
                        mask_idx = [x + start for x in mask_idx]
                        input_ids[choice_id][start: end] = torch.tensor([random.choice(range(len(self.tokenizer))) if _idx in mask_idx else input_ids[choice_id][_idx] for _idx in range(start, start+added_length)]).to(input_ids.device)
                    else:
                        start, end, _ = input_ids_indices[choice_id]
                        added_length = end - start
                        mask_idx = random.sample(range(added_length), int(added_length * self.args.replace_ratio))
                        mask_idx = [x + start for x in mask_idx]
                        attention_mask[choice_id][start : end] = torch.tensor([0 if _idx in mask_idx else 1 for _idx in range(start, start+added_length)]).to(attention_mask.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits                             # (B * num_templates, seq_len, vocab_size)
        batchid, indices = torch.where(input_ids == self.tokenizer.mask_token_id) # See how the word is inserted

        if is_causal_model(self.args.model):
            # it predicts next word
            indices = indices -1

        mask_logits = logits[batchid, indices,:]         # (B * num_templates, vocab_size)
        label_words_logits = mask_logits
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

        query_logits = self.get_logits(input_ids, outputs=outputs) # (batch_size, vocab_size)
        label_words_logits = query_logits[:, self.label_word_ids]    # (batch_size, num_candidates)
        query_logits = torch.softmax(query_logits, dim=-1) # (batch_size, vocab_size)
        label_words_logits = torch.softmax(label_words_logits, dim=-1) # (batch_size, num_candidates)

        # Directly return the logits
        # kl_dists = self.anchor_store.knn_infer(query_logits) # [B, K+1]
        # scaled_dists = -1.0 / self.knn_T * kl_dists

        # top_dists, top_indices = torch.topk(scaled_dists, self.knn_k) # [B, K+1], [B, K+1]
        # new_vals = values.unsqueeze(0).repeat(self.args.batch_size, 1) # [B, L]
        # top_values = torch.gather(new_vals, 1, top_indices).unsqueeze(-1)  # [B, K, 1]
        # knn_weight = torch.softmax(top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]
        
        # # init knn-prob
        # knn_tgt_prob = torch.zeros([self.args.batch_size, self.knn_k, self.args.num_labels], dtype=torch.float32, device=keys.device)
        # knn_tgt_prob.scatter_(2, top_values, knn_weight) # The scatter function is used to scatter the values in knn_weight to the corresponding positions in knn_tgt_prob. 
        # # The 2 in the first parameter means that the scatter is performed on the third dimension of knn_tgt_prob, and the second parameter is the index of the position to be scattered. The third parameter is the value to be scattered.
        # # the final dimension is [B, K, V]
        # prob = knn_tgt_prob.sum(dim=-2)  # [B, V]
        prob = self.anchor_store.knn_calibrate(query_logits) # [B, V]

        # softmax the last dimension
        prob = torch.softmax(prob, dim=-1)

        prob = self.args.beta * prob + (1-self.args.beta) * label_words_logits
        return prob

        return label_words_logits
