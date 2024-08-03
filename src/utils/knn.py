import string

import os
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
import copy
import random
from torch_scatter import scatter
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, BertTokenizerFast, BertForMaskedLM, BertModel
import torch.optim as optim
from torch.autograd import Variable
from textattack.models.wrappers import ModelWrapper
import transformers
from tqdm import trange

class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        batch_idx = torch.arange(labels.shape[0], device=logits.device)
        loss = log_probs[batch_idx, labels]
        n = logits.shape[-1] - 1.0
        p = 1.0 - self.smoothing
        q = self.smoothing / n
        sum_probs = torch.sum(log_probs, dim=-1)
        loss = p * loss + q * (sum_probs - loss)
        return -loss.sum()

'''
feature mapping + compression
'''
class MetaKNetwork(nn.Module):
    def __init__(self, input_size, map_size, output_size):
        super().__init__()
        self.f1_func = nn.Linear(input_size, 32)
        self.f2_func = nn.Linear(32, output_size)     # select_topk
        self.dropout = nn.Dropout(p=0.3)
        self.loss = SmoothedCrossEntropyLoss(smoothing=0.1)
        self.reset_parameters()

    def forward(self, x):
        hidden = torch.relu(self.f1_func(self.dropout(x)))
        select_topk_prob = torch.softmax(self.f2_func(self.dropout(hidden)), dim=-1)
        return select_topk_prob

    def reset_parameters(self):
        # f1_func
        nn.init.kaiming_uniform_(self.f1_func.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.f1_func.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f1_func.bias, -bound, bound)
        # f2_func
        nn.init.kaiming_uniform_(self.f2_func.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.f2_func.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f2_func.bias, -bound, bound)

class CompactLayer(nn.Module):
    def __init__(self, hidden_size, map_size):
        super().__init__()
        self.f1_func = nn.Linear(hidden_size, map_size)
        self.dropout = nn.Dropout(p=0.3)
        self.loss = SmoothedCrossEntropyLoss(smoothing=0.1)
        self.reset_parameters()

    def forward(self, x):
        hidden = torch.relu(self.f1_func(self.dropout(x)))
        return hidden

    def reset_parameters(self):
        # f1_func
        nn.init.kaiming_uniform_(self.f1_func.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.f1_func.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f1_func.bias, -bound, bound)

def load_dataset(file_name, sampleNum=0):
    data_inputs = []
    data_labels = []
    with open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            items = line.strip().split('\t')
            # for index in range(sampleNum):
            data_inputs.append(items[sampleNum])
            data_labels.append(int(items[-1]))

    return data_inputs, data_labels

class HuggingFaceSentimentAnalysisPipelineWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, tokenizer, model, model_dir, dataset, task, data_dir, hidden_size, \
            batch_size, train_epoch, tindex, shot, \
            map_size=32, prompt_num=2, num_labels=2, knn_k=8, knn_T=5, ensemble_num=1, sampled_num=1, max_length=512, device='cuda',mode=0):
        if "roberta" in model_dir:
            MASK_TOKEN = "<mask>"
            SEP_TOEKN = "</s>"
        elif "bert" in model_dir:
            MASK_TOKEN = "[MASK]"
            SEP_TOEKN = "[SEP]"

        
        self.device = device
        self.mask_token = MASK_TOKEN
        self.sep_token = SEP_TOEKN
        self.tokenizer = tokenizer
        self.model = model
        self.model_dir = model_dir
        self.ensemble_num = ensemble_num
        self.sampled_num = sampled_num
        self.dataset = dataset if dataset != 'sst-2' else 'sst2'
        self.task = task if task != 'sst2' else 'sst-2'
        self.data_dir = data_dir
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.prompt_num = prompt_num
        self.knn_k = knn_k
        self.knn_T = knn_T
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_epoch = train_epoch
        self.map_size = map_size
        self.tindex = tindex
        self.shot = shot
        self.mode = mode

        compactLayer, train_ds_inputs, train_labels, valid_ds_inputs, valid_labels, get_vec_fun, label_word_ids, demon_sample = self.pre_experiment(tokenizer, model)
        if self.mode == 1:
            metakLayer, valid_acc = self.model_training(train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
                map_size, map_size, batch_size, train_epoch, knn_T, knn_k, num_labels,
                sampled_num, ensemble_num, 1)
            metakLayer.eval()
        else:
            metakLayer = None

        self.compactLayer = compactLayer
        self.metakLayer = metakLayer
        self.train_ds_inputs = train_ds_inputs
        self.valid_ds_inputs = valid_ds_inputs
        self.get_vec_fun = get_vec_fun
        self.label_word_ids = label_word_ids 
        self.demon_sample = demon_sample
        
    def outs_to_logits(self, text_inputs):
        return self.test(text_inputs, self.compactLayer, self.train_ds_inputs, self.valid_ds_inputs, self.get_vec_fun, self.label_word_ids, self.demon_sample)
    
    def __call__(self, text_inputs, attack=True):
        # batch of testing inputs, convert into required demon format
        # build datastore based on them
        # calculate the top-k probability
        if attack:
            raw_outputs = self.test(text_inputs, self.compactLayer, self.train_ds_inputs, self.valid_ds_inputs, self.get_vec_fun, self.label_word_ids, self.demon_sample)
            outputs = []
            for output in raw_outputs:
                # convert output from numpy array to list
                outputs.append(output.tolist())
            return np.array(outputs)
        else:
            all_seeds = [13]
            seed = all_seeds[0]
            test_inputs, test_labels = load_dataset(self.data_dir + '/' + '{0}/basic/{1}/{2}-{3}.tindex{4}.test'.format(self.dataset, self.model_dir, self.shot, seed, self.tindex), self.ensemble_num)

            save_path = '{0}.shot{1}.seed{2}.tindex{3}.test.s{4}'.format(self.dataset, self.shot, seed, self.tindex, self.sampled_num)

            save_path = os.path.join(self.data_dir, 'knn_datastore', save_path)
            test_ds_inputs = self.load_datastore(save_path, test_inputs, test_labels, \
                self.batch_size, self.hidden_size, self.get_vec_fun, label_word_ids=self.label_word_ids, reuse=False)
            with torch.no_grad():
                test_ds_inputs = (self.compactLayer(test_ds_inputs[0]), test_ds_inputs[1], test_ds_inputs[2])
            # evaluate 
            test_acc = self.evaluate_dataset(self.train_ds_inputs, test_ds_inputs, test_labels, self.metakLayer, \
                    self.knn_T, self.knn_k, self.batch_size, self.num_labels, self.ensemble_num, True if self.mode == 1 else False)
            
            print('Test acc: ', test_acc)
            return test_acc
    def load_datastore(self, save_path, train_inputs, train_labels, batch_size,
                hidden_size, get_vec_fun, label_word_ids=None, reuse=True):
        if (save_path is not None) and (os.path.exists(save_path + '.keys.npy') and reuse):
            datastore_keys = np.load(save_path + '.keys.npy')
            datastore_vals = np.load(save_path + '.labels.npy')
            if label_word_ids is not None:
                datastore_probs = np.load(save_path + '.probs.npy')
        else:
            datastore_keys, datastore_vals, datastore_probs = self.build_datastore(train_inputs, \
                train_labels, batch_size, hidden_size, get_vec_fun, label_word_ids)
            if save_path is not None:
                np.save(save_path + '.keys', datastore_keys) 
                np.save(save_path + '.labels', datastore_vals) 
                if label_word_ids is not None:
                    np.save(save_path + '.probs', datastore_probs) 
        
        datastore_keys = torch.from_numpy(datastore_keys).to(self.device)
        datastore_vals = torch.from_numpy(datastore_vals).to(self.device)
        if label_word_ids is not None:
            datastore_probs = torch.from_numpy(datastore_probs).to(self.device)

        return datastore_keys, datastore_vals, datastore_probs if label_word_ids is not None else None


    def get_results_with_prompt(self, sents, model, tokenizer, max_length, label_word_ids):
        with torch.no_grad():
            inputs = tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs['input_ids'] = inputs['input_ids'].to(self.device) # shape: (sentence_num, sequence_length)
            # inputs['token_type_ids'] = inputs['token_type_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
            inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            
            results = model(**inputs, output_hidden_states=True) # shape: (sentence_num, sequence_length, hidden_size)

            hidden_states = results.hidden_states[-1] # last layer hidden state shape: (sentence_num, sequence_length, hidden_size)
            # if label_word_ids is not None:
            logits = results.logits
            label_idx = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)  # (sentence_num, sequence_length)

            output_hidden_state = torch.zeros([len(sents), hidden_states.size()[-1]], dtype=torch.float32, device=hidden_states.device) # (sentence_num, hidden_size)
            # if label_word_ids is not None:
            output_logits = torch.zeros([len(sents), len(label_word_ids)], dtype=torch.float32, device=hidden_states.device)
            for index in range(len(label_idx[0])):
                index_0 = label_idx[0][index]
                index_1 = label_idx[1][index]
                output_hidden_state[index,:] = hidden_states[index_0, index_1, :]
                # if label_word_ids is not None:
                for word_index, word_id in enumerate(label_word_ids):
                    output_logits[index, word_index] = logits[index_0, index_1, word_id]
            return output_hidden_state, torch.softmax(output_logits, dim=-1) if label_word_ids is not None else None

    def build_datastore(self, data_inputs, data_labels, batch_size, hidden_size, func_get_vec, label_word_ids=None):
        
        total_len = len(data_labels)
        datastore_keys = np.zeros([total_len, hidden_size], dtype=np.float32)
        datastore_vals = np.zeros([total_len], dtype=np.int64)
        if label_word_ids is not None:
            datastore_probs = np.zeros([total_len, len(label_word_ids)], dtype=np.float32)

        for start in range(0, total_len, batch_size):
            end = min(total_len, start + batch_size)
            vecs, probs = func_get_vec(data_inputs[start:end], label_word_ids)
            datastore_keys[start:end] = vecs.cpu().numpy()
            datastore_vals[start:end] = data_labels[start:end]
            if label_word_ids is not None:
                datastore_probs[start:end] = probs.cpu().numpy()

        return datastore_keys, datastore_vals, datastore_probs if label_word_ids is not None else None

    def calculate_topk_prob(self, queries, keys, values, knn_T, knn_k, batch_size, 
                num_labels, isTrain=False, isRemoveTop1=False):
        # queries [B, H]
        # keys [L, H]
        
        kl_dists = torch.mean(keys.unsqueeze(0) * (keys.unsqueeze(0) - queries.unsqueeze(1)), dim=-1) # [B, L]
        dists = ((keys.unsqueeze(0) - queries.unsqueeze(1)) ** 2).sum(-1) # [B, L]
        
        scaled_dists = -1.0 / knn_T * dists # [B, L]
        top_dists, top_indices = torch.topk(scaled_dists, (knn_k + 1) if isRemoveTop1 else knn_k) # [B, K+1], [B, K+1]
        new_vals = values.unsqueeze(0).repeat(batch_size, 1) # [B, L]
        top_values = torch.gather(new_vals, 1, top_indices[:, 1:] if isRemoveTop1 else top_indices).unsqueeze(-1)  # [B, K, 1]
        knn_weight = torch.softmax(top_dists[:, 1:] if isRemoveTop1 else top_dists, dim=-1).unsqueeze(-1)  # [B, K, 1]
        
        # init knn-prob
        knn_tgt_prob = torch.zeros([batch_size, knn_k, num_labels], dtype=torch.float32, device=keys.device)
        if isTrain:
            knn_tgt_prob = Variable(knn_tgt_prob).clone()
        knn_tgt_prob.scatter_(2, top_values, knn_weight) # The scatter function is used to scatter the values in knn_weight to the corresponding positions in knn_tgt_prob. 
        # The 2 in the first parameter means that the scatter is performed on the third dimension of knn_tgt_prob, and the second parameter is the index of the position to be scattered. The third parameter is the value to be scattered.
        # the final dimension is [B, K, V]
        prob = knn_tgt_prob.sum(dim=-2)  # [B, V]

        return prob

    def calculate_adaptive_topk_prob(self, queries, queries_probs, keys, values, knn_T,
                knn_k_list, batch_size, num_labels, isRemoveTop1=False):
        # queries [B, H]
        # keys [L, H]
        dists = ((keys.unsqueeze(0) - queries.unsqueeze(1)) ** 2).sum(-1) # [B, L]
        scaled_dists = -1.0 / knn_T * dists
        # print(len(scaled_dists), knn_k_list)
        top_dists, top_indices = torch.topk(scaled_dists, 
            knn_k_list[-1]) # [B, K_max]
        new_vals = values.unsqueeze(0).repeat(batch_size, 1)
        top_values = torch.gather(new_vals, 1, top_indices) # [B, K_max]
        if isRemoveTop1:
            top_dists = top_dists[:, 1:] # [B, K_max]

        knn_prob_list = []
        for topk in knn_k_list:
            if topk == 0:
                knn_prob_list.append(queries_probs)
            else:
                knn_weight_k = torch.softmax(top_dists[:, :topk], dim=-1).unsqueeze(-1)  # [B, K, 1]
                top_values_k = top_values[:, :topk].unsqueeze(-1) # [B, K, 1]
                knn_tgt_prob = torch.zeros([batch_size, topk, num_labels], dtype=torch.float32, device=queries.device)
                knn_tgt_prob.scatter_(2, top_values_k, knn_weight_k) # [B, K, V]
                knn_prob = knn_tgt_prob.sum(dim=-2)  # [B, V]
                knn_prob_list.append(knn_prob)
        
        knn_prob = torch.stack(knn_prob_list, dim=1) # [B, k_num, V]

        # count label number for top-k
        top_values_cpu = top_values.cpu()
        top_values_count = torch.zeros_like(top_values_cpu) 
        for index in range(batch_size):
            value_set = set()
            for k_index in range(knn_k_list[-1]):
                value_set.add(top_values[index, k_index])
                top_values_count[index, k_index] = len(value_set)
        top_values_count = top_values_count.to(self.device) # [B, K]

        selected_topk_dists = torch.zeros([batch_size, len(knn_k_list)], dtype=torch.float32, device=queries.device)
        selected_topk_count = torch.zeros([batch_size, len(knn_k_list)], dtype=torch.float32, device=queries.device)
        for index, topk in enumerate(knn_k_list):
            selected_topk_dists[:,index] = top_dists[:,topk - 1 if topk != 0 else topk]
            selected_topk_count[:,index] = top_values_count[:,topk - 1 if topk != 0 else topk]
        
        return knn_prob, selected_topk_dists, selected_topk_count

    def evaluate_dataset(self, train_ds_inputs, test_ds_inputs, test_labels, metakLayer, 
            knn_T, knn_k, batch_size, num_labels, ensemble_num, isAdaptive=True):

        
        topk_list = [i * 4 for i in range(self.shot // 4 + 1)]

        train_ds_keys, train_ds_vals, train_ds_probs = train_ds_inputs
        test_ds_keys, test_ds_vals, test_ds_probs = test_ds_inputs
        total_len = len(test_labels)
        correct = 0
        for start in trange(0, total_len, batch_size * ensemble_num):
            with torch.no_grad():
                end = min(total_len, start + batch_size * ensemble_num)
                cur_batch = (end - start) // ensemble_num
                test_ids = torch.Tensor([test_labels[index] for index in range(start, end, ensemble_num)]).to(self.device)
                test_vecs = test_ds_keys[start:end]
                if isAdaptive:
                    test_probs = test_ds_probs[start:end]
                    knn_probs, top_dists, top_values_count = self.calculate_adaptive_topk_prob(test_vecs, test_probs, \
                        train_ds_keys, train_ds_vals, knn_T, topk_list, end - start, num_labels)
                    input_feature = torch.hstack((top_dists, top_values_count)) # [B, 2 * len(topk_list)]
                    adaptive_weights = metakLayer(input_feature).unsqueeze(-1)
                    knn_probs = (knn_probs * adaptive_weights).sum(dim=1)
                else:
                    knn_probs = self.calculate_topk_prob(test_vecs, train_ds_keys, train_ds_vals, \
                            knn_T, knn_k, end - start, num_labels)
                if ensemble_num > 1:
                    knn_probs = knn_probs.reshape([cur_batch, -1, num_labels])
                    total_probs = torch.mean(knn_probs, dim=-2)
                else:
                    total_probs = knn_probs
                predict_ids = total_probs.argmax(axis=-1)
                correct += torch.eq(predict_ids, test_ids).int().sum().cpu().numpy()      

        return 1.0 * correct * ensemble_num / total_len

    def get_logits(self, train_ds_inputs, test_ds_inputs, metakLayer, knn_T, knn_k, batch_size, num_labels, ensemble_num, isAdaptive=True):
        topk_list = [0, 4, 8, 12, 16]
        topk_list = [i * 4 for i in range(self.shot // 4 + 1)]

        # topk_list = [0, 4, 8, 12]
        train_ds_keys, train_ds_vals, train_ds_probs = train_ds_inputs
        test_ds_keys, test_ds_vals, test_ds_probs = test_ds_inputs
        total_len = len(test_ds_keys)
        correct = 0
        for start in range(0, total_len, batch_size * ensemble_num):
            with torch.no_grad():
                end = min(total_len, start + batch_size * ensemble_num)
                cur_batch = (end - start) // ensemble_num
                test_vecs = test_ds_keys[start:end]
                if isAdaptive:
                    test_probs = test_ds_probs[start:end]
                    knn_probs, top_dists, top_values_count = self.calculate_adaptive_topk_prob(test_vecs, test_probs, \
                        train_ds_keys, train_ds_vals, knn_T, topk_list, end - start, num_labels)
                    input_feature = torch.hstack((top_dists, top_values_count))
                    adaptive_weights = metakLayer(input_feature).unsqueeze(-1)
                    knn_probs = (knn_probs * adaptive_weights).sum(dim=1)
                else:
                    knn_probs = self.calculate_topk_prob(test_vecs, train_ds_keys, train_ds_vals, \
                            knn_T, knn_k, end - start, num_labels)
                if ensemble_num > 1:
                    knn_probs = knn_probs.reshape([cur_batch, -1, num_labels])
                    total_probs = torch.mean(knn_probs, dim=-2)
                else:
                    total_probs = knn_probs
        return total_probs # [B, Label]

    def model_training(self, train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
            hidden_size, map_size, batch_size, total_epoch, knn_T, knn_k, num_labels,
            sampled_num, ensemble_num=1, train_mode=0):

        topk_list = [0, 4, 8, 12, 16]
        topk_list = [i * 4 for i in range(self.shot // 4 + 1)]

        # topk_list = [0, 4, 8, 12]
        train_ds_keys, train_ds_vals, train_ds_probs = train_ds_inputs
        if train_mode == 0:
            trainLayer = CompactLayer(hidden_size, map_size).to(self.device)
        else:
            trainLayer = MetaKNetwork(len(topk_list) * 2, map_size, len(topk_list)).to(self.device)

        total_len = len(train_labels)
        # split training set
        train_number = total_len // sampled_num
        total_len = total_len // 2 # haif for buiding datastore and rest for training

        searched_ds_keys = torch.zeros([total_len, hidden_size], dtype=torch.float32, device=train_ds_keys.device)
        searched_ds_vals = torch.zeros([total_len], dtype=torch.int64, device=train_ds_keys.device)
        query_ds_keys = torch.zeros([total_len, hidden_size], dtype=torch.float32, device=train_ds_keys.device)
        query_ds_probs = torch.zeros([total_len, num_labels], dtype=torch.float32, device=train_ds_keys.device)
        query_ds_vals = torch.zeros([total_len], dtype=torch.int64, device=train_ds_keys.device)

        class_tag = [1 - train_mode] * num_labels
        searched_index = 0
        query_index = 0
        for index in range(train_number):
            cur_index = index * sampled_num
            class_id = train_labels[cur_index]
            if class_tag[class_id] == 0:
                searched_ds_keys[searched_index:searched_index+sampled_num] = train_ds_keys[cur_index:cur_index+sampled_num]
                searched_ds_vals[searched_index:searched_index+sampled_num] = train_ds_vals[cur_index:cur_index+sampled_num]
                searched_index += sampled_num
                class_tag[class_id] = 1
            else:
                query_ds_keys[query_index:query_index+sampled_num] = train_ds_keys[cur_index:cur_index+sampled_num]
                query_ds_probs[query_index:query_index+sampled_num] = train_ds_probs[cur_index:cur_index+sampled_num]
                query_ds_vals[query_index:query_index+sampled_num] = train_ds_vals[cur_index:cur_index+sampled_num]
                query_index += sampled_num
                class_tag[class_id] = 0

        best_Layer = None
        best_valid_acc = 0
        optimizer = optim.Adam(trainLayer.parameters())

        for epoch in range(total_epoch):
            # shuffle data
            running_loss = 0.0
            trainLayer.train()
            training_order = torch.randperm(total_len)
            all_input_keys = query_ds_keys[training_order]
            all_input_probs = query_ds_probs[training_order]
            all_input_vals = query_ds_vals[training_order]
            
            for start in range(0, total_len, 64):
                end = min(total_len, start + 64)
                # build train data
                input_vecs = all_input_keys[start:end]
                input_probs = all_input_probs[start:end]
                input_vals = all_input_vals[start:end]
                # zero the parameter gradients
                optimizer.zero_grad()
                if train_mode == 0:
                    cur_topk = 8
                    input_vecs = trainLayer(input_vecs)
                    all_keys = trainLayer(searched_ds_keys)
                    knn_probs = self.calculate_topk_prob(input_vecs, all_keys, searched_ds_vals, knn_T, cur_topk, \
                            end - start, num_labels, True)
                    loss = trainLayer.loss(torch.log(knn_probs + 1e-20), input_vals)
                else:
                    knn_probs, top_dists, top_values_count = self.calculate_adaptive_topk_prob(input_vecs, input_probs, \
                            searched_ds_keys, searched_ds_vals, knn_T, topk_list, end - start, num_labels)
                    input_feature = torch.hstack((top_dists, top_values_count))
                    adaptive_weights = trainLayer(input_feature).unsqueeze(-1)
                    knn_probs = (knn_probs * adaptive_weights).sum(dim=1)
                    loss = trainLayer.loss(torch.log(knn_probs + 1e-20), input_vals)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainLayer.parameters(), 10)
                optimizer.step()
                running_loss += loss.item()

            # valid 
            trainLayer.eval()
            if train_mode == 0:
                with torch.no_grad():
                    train_ds_inputs_new = (trainLayer(train_ds_inputs[0]), train_ds_inputs[1], train_ds_inputs[2])
                    valid_ds_inputs_new = (trainLayer(valid_ds_inputs[0]), valid_ds_inputs[1], valid_ds_inputs[2])
            else:
                train_ds_inputs_new = train_ds_inputs
                valid_ds_inputs_new = valid_ds_inputs
            valid_acc = self.evaluate_dataset(train_ds_inputs_new, valid_ds_inputs_new, valid_labels, trainLayer, \
                            knn_T, knn_k, batch_size, num_labels, 1, False if train_mode == 0 else True)

            if valid_acc > best_valid_acc:
                best_Layer = copy.deepcopy(trainLayer)
                best_valid_acc = valid_acc

        return best_Layer, best_valid_acc

    def pre_experiment(self, tokenizer, base_model):
        # load pre-trained model
        # tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        model = base_model
        print(base_model)
        # tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
        # model = BertModel.from_pretrained(args.model_dir).to(DEVICE)

        model.eval()

        # all_seeds = [13, 21, 42, 87, 100]
        all_seeds = [13]

        # shots = [16, 32, 64]
        # shots = [16]
        ## creating datastore
        # build map function
        def get_vec_fun(sents, label_word_ids=None):
            return self.get_results_with_prompt(sents, model, tokenizer, self.max_length, label_word_ids)

        test_ensemble_num_list = [self.ensemble_num if self.ensemble_num > 1 else 1]
        valid_std_fm = np.zeros(self.prompt_num * len(all_seeds) * 4, dtype=np.float32)
        valid_std_all = np.zeros(self.prompt_num * len(all_seeds) * 4, dtype=np.float32)
        test_std_fm_list = [0] * len(test_ensemble_num_list)
        test_std_all_list = [0] * len(test_ensemble_num_list)
        test_std_bae_all_list = [0] * len(test_ensemble_num_list)
        test_std_textfooler_all_list = [0] * len(test_ensemble_num_list)
        for index in range(len(test_ensemble_num_list)):
            test_std_fm_list[index] = np.zeros(self.prompt_num * len(all_seeds) * 4, dtype=np.float32)
            test_std_all_list[index] = np.zeros(self.prompt_num * len(all_seeds) * 4, dtype=np.float32)
            test_std_bae_all_list[index] = np.zeros(self.prompt_num * len(all_seeds) * 4, dtype=np.float32)
            test_std_textfooler_all_list[index] = np.zeros(self.prompt_num * len(all_seeds) * 4, dtype=np.float32)

        task_index = 0
        seed = all_seeds[0]
        # for seed in all_seeds:
        # load datasets
        train_inputs, train_labels = load_dataset(self.data_dir + '/' + '{0}/basic/{1}/{2}-{3}.tindex{4}.train'.format(self.dataset, self.model_dir, self.shot, seed, self.tindex), self.sampled_num)
        valid_inputs, valid_labels = load_dataset(self.data_dir + '/' + '{0}/basic/{1}/{2}-{3}.tindex{4}.valid'.format(self.dataset, self.model_dir, self.shot, seed, self.tindex), self.ensemble_num)
        demon_sample = train_inputs[0].split(self.sep_token)[:2]

        # print(train_inputs)
        # tindex = 0
        # load the verbalizers
        label_word_ids = get_verbalizers_ids(self.task, self.tindex, tokenizer)

        # print('Train size: ', len(train_inputs))
        # print('Label size: ', len(label_word_ids))
        # print(label_word_ids)

        # TODO: Toggle the train seed!
        train_seed = 1
        np.random.seed(train_seed)
        random.seed(train_seed)
        torch.manual_seed(train_seed)
        
        save_path = '{0}.shot{1}.seed{2}.tindex{3}.train.s{4}'.format(self.dataset, self.shot, seed, self.tindex, self.sampled_num)
        if not os.path.exists(os.path.join(self.data_dir, 'knn_datastore')):
            os.makedirs(os.path.join(self.data_dir, 'knn_datastore'))

        save_path = os.path.join(self.data_dir, 'knn_datastore', save_path)
        train_ds_inputs = self.load_datastore(save_path, train_inputs, train_labels, \
            self.batch_size, self.hidden_size, get_vec_fun, label_word_ids=label_word_ids, reuse=False)

        # print('Train datastore size: ', len(train_ds_inputs[0]))

        save_path = '{0}.shot{1}.seed{2}.tindex{3}.valid.s{4}'.format(self.dataset, self.shot, seed, self.tindex, self.ensemble_num)
        save_path = os.path.join(self.data_dir, 'knn_datastore', save_path)
        valid_ds_inputs = self.load_datastore(save_path, valid_inputs, valid_labels, \
                self.batch_size, self.hidden_size, get_vec_fun, label_word_ids=label_word_ids, reuse=False)

        # print('Valid datastore size: ', len(valid_ds_inputs[0]))
        # model training
        compactLayer, valid_acc_fm = self.model_training(train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
                self.hidden_size, self.map_size, self.batch_size, self.train_epoch, self.knn_T, self.knn_k, self.num_labels,
                self.sampled_num, self.ensemble_num, 0)
        print('Train finished!')
        compactLayer.eval()
        valid_std_fm[task_index] = valid_acc_fm

        # build new feature function
        with torch.no_grad():
            train_ds_inputs = (compactLayer(train_ds_inputs[0]), train_ds_inputs[1], train_ds_inputs[2])
            valid_ds_inputs = (compactLayer(valid_ds_inputs[0]), valid_ds_inputs[1], valid_ds_inputs[2])

        # metakLayer, valid_acc = self.model_training(train_ds_inputs, train_labels, valid_ds_inputs, valid_labels,
        #     self.map_size, self.batch_size, self.train_epoch, self.knn_T, self.knn_k, self.num_labels,
        #     self.sampled_num, self.ensemble_num, 1)
        # metakLayer.eval()
        # valid_std_all[task_index] = valid_acc

        return compactLayer, train_ds_inputs, train_labels, valid_ds_inputs, valid_labels, get_vec_fun, label_word_ids, demon_sample

    def test(self, raw_test_inputs, compactLayer, train_ds_inputs, valid_ds_inputs, get_vec_fun, label_word_ids, demon_sample):
        # convert the input into required format
        demon_inputs = demon_sample
        test_input, test_labels = raw_test_inputs, [1] * len(raw_test_inputs)

        test_input = [get_prompt_str(test_input[index], 0, self.task, self.tindex, [self.mask_token]) for index in range(len(test_input))]
        test_input_final = []
        for index in range(len(test_input)):
            random_index = np.random.randint(0, len(demon_inputs))
            test_input_final.append(demon_inputs[random_index] + f' {self.sep_token}' + demon_inputs[1 - random_index] + f' {self.sep_token}' + test_input[index])
        test_ds_inputs = self.load_datastore(None, test_input_final, test_labels, self.batch_size, \
                                        self.hidden_size, get_vec_fun, label_word_ids, True)
        with torch.no_grad():
            test_ds_inputs = (compactLayer(test_ds_inputs[0]), test_ds_inputs[1], test_ds_inputs[2])
        
        results = self.get_logits(train_ds_inputs, test_ds_inputs, self.metakLayer, \
                            self.knn_T, self.knn_k, self.batch_size, self.num_labels, 1, False if self.mode == 0 else True)
        return results

# we set verbalizers to the list of '<mask>' when using prompt
def get_prompt_str(input_str, label_index, task, tindex, verbalizers, mask_token='<mask>', sep_token='</s>'):
    output_str = ''

    if task in ['mr', 'cr', 'SST-2', 'sst-5', 'sst-2', 'sst2']:
        if tindex == 0:
            output_str = '{0}. A{1} one.'.format(input_str, verbalizers[label_index]) 
            # output_str = '{0} Sentiment of the statement is{1}'.format(input_str, verbalizers[label_index])
        elif tindex == 1:
            output_str = '{0}. It was{1}.'.format(input_str, verbalizers[label_index]) 
            # output_str = '{0}</s>{1}'.format(input_str, verbalizers[label_index])
        elif tindex == 2:
            # output_str = '{0} All in all{1}.'.format(input_str, verbalizers[label_index])
            output_str = '{0}. This is a{1} statement.'.format(input_str, verbalizers[label_index])
        elif tindex == 3:
            # output_str = '{0} A{1} piece.'.format(input_str, verbalizers[label_index]) 
            output_str = '{0}{1}The statement is{2}'.format(input_str, sep_token, verbalizers[label_index])
        elif tindex == 4:
            output_str = '{0} Sentiment of the statement is{1}'.format(input_str, verbalizers[label_index])
        elif tindex == 5:
            output_str = '{0}{1}{2}'.format(input_str, sep_token, verbalizers[label_index])
        elif tindex == 6:
            output_str = '{0}. All in all{1}.'.format(input_str, verbalizers[label_index])
        elif tindex == 7:
            output_str = '{0} A{1} piece.'.format(input_str, verbalizers[label_index])
    elif task == 'subj':
        if tindex == 0:
            output_str = '{0} This is{1}.'.format(input_str, verbalizers[label_index]) 
        elif tindex == 1:
            output_str = '{0} It\'s all{1}.'.format(input_str, verbalizers[label_index]) 
        elif tindex == 2:
            output_str = '{0} It\'s{1}.'.format(input_str, verbalizers[label_index]) 
        elif tindex == 3:
            output_str = '{0} Is it{1}?'.format(input_str, verbalizers[label_index]) 
    elif task == 'trec':
        if tindex == 0:
            output_str = '{0}{1}:'.format(input_str, verbalizers[label_index]) 
        elif tindex == 1:
            output_str = '{0} Q:{1}:'.format(input_str, verbalizers[label_index])
        elif tindex == 2:
            output_str = '{0} why{1}?'.format(input_str, verbalizers[label_index])
        elif tindex == 3:
            output_str = '{0} Answer:{1}.'.format(input_str, verbalizers[label_index])
    elif task in ['rte', 'cb']:   ### input: (premise, hypothesis)
        if tindex == 0:
            output_str = '"{1}"?{2}, "{0}"'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
        elif tindex == 1:
            output_str = '{1}?{2}, {0}'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
        elif tindex == 2:
            output_str = '"{1}"?{2}. "{0}"'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
        elif tindex == 3:
            output_str = '{1}?{2}. {0}'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
        elif tindex == 4:
            if task == 'rte':
                output_str = '{0} question: {1} True or False? answer:{2}.'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
            else:
                output_str = '{0} question: {1} true, false or neither? answer:{2}.'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
    elif task == 'wic':     ### input: (sentecne1, sentence2, word)
        if tindex == 0:
            output_str = '"{0}" / "{1}" Similar sense of "{2}"?{3}.'.format(input_str[0], input_str[1], input_str[2], verbalizers[label_index])
        elif tindex == 1:
            output_str = '{0} {1} Does {2} have the same meaning in both sentences?{3}.'.format(input_str[0], input_str[1], input_str[2], verbalizers[label_index])
        elif tindex == 2:
            output_str = '{2} . Sense (1) (a) "{0}" ({3}) "{1}"'.format(input_str[0], input_str[1], input_str[2], verbalizers[label_index])
    elif task == 'qnli':    ### input: (question, sentence)
        if tindex < 2:
            output_str = '{1}. Question: {0}? Answer:{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
        elif tindex < 4:
            output_str = '{1}. Based on the previous sentence, {0}?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
        else:
            output_str = 'Based on the following sentence, {0}?{2}. {1}'.format(input_str[0], input_str[1], verbalizers[label_index])
    elif task == 'qqp':   ### input: (question1, question2)
        if tindex < 2:
            output_str = 'Do "{0}" and "{1}" have the same meaning?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
        elif tindex < 4:
            output_str = '{0}. Based on the previous question, {1}?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
        else:
            output_str = 'Based on the following question, {0}?{2}. {1}'.format(input_str[0], input_str[1], verbalizers[label_index])
    elif task == 'mrpc':  ### input: (sentence1, sentence2)
        if tindex < 2:
            output_str = 'Do "{0}" and "{1}" have the same meaning?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
        elif tindex < 4:
            output_str = '{0}. Based on the previous sentence, {1}?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
        else:
            output_str = 'Based on the following sentence, {0}?{2}. {1}'.format(input_str[0], input_str[1], verbalizers[label_index])

    return output_str

# For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
def get_verbalizers_ids(task, tindex, tokenizer):
    word_list = get_verbalizers_str(task, tindex)

    return [tokenizer(word, add_special_tokens=False)['input_ids'][0] for word in word_list]

# For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
def get_verbalizers_str(task, tindex):
    if task in ['mr', 'cr', 'SST-2', 'sst-2', 'sst2']:
        word_list = [" terrible", " great"]
        if tindex > 3:
            word_list = [" negative", " positive"]
    elif task == 'sst-5':
        word_list = [" terrible", " bad", " okay", " good", " great"]
    elif task == 'subj':
        word_list = [" subjective", " objective"]
    elif task == 'trec':
        word_list = [" Description", " Entity", " Expression", " Human", " Location", " Number"]
    elif task == 'rte':
        if tindex == 4:
            word_list = [" true", " false"]
        word_list = [" Yes", " No"]
    elif task == 'cb':
        if tindex == 4:
            word_list = [' true', ' false', ' neither']
        word_list = [" Yes", " No", " Maybe"]
    elif task == 'wic':
        if tindex == 2:
            word_list = ["2", "b"]
        word_list = [" No", " Yes"]
    elif task == 'qnli':
        if tindex in [0, 2, 4]:
            word_list = [" Yes", " No"]
        word_list = [" true", " false"]
    elif task in ['qqp', 'mrpc']:
        if tindex in [0, 2, 4]:
            word_list = [" No", " Yes"]
        word_list = [" false", " true"]
    else:
        assert "error task name....."

    return word_list