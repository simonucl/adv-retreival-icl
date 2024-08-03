import os 
import sys
import random
import argparse
import string 
import random
import editdistance
import json
import textattack
sys.path.append('../')
from knn.utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing Data for KNN-C")

    parser.add_argument("--train", type=str, help="train data")
    parser.add_argument("--valid", type=str, help="valid data")
    parser.add_argument("--test", type=str, help="test data")
    parser.add_argument("--output", type=str, help="output path")
    parser.add_argument("--task", type=str, help="the task name")
    parser.add_argument("--mode", type=int, help="0:basic, 1:prompt, 2:prompt+demon")
    parser.add_argument("--tindex", type=int, default=0, help="the templete index for prompt")
    parser.add_argument("--num-label", type=int, default=2, help="the number of label for different tasks")
    parser.add_argument("--num-demon-train", type=int, default=16, help="the number of demonstration")
    parser.add_argument("--num-demon-test", type=int, default=16, help="the number of demonstration")
    parser.add_argument("--seed", type=int, default=53, help="the number of demonstration")
    parser.add_argument("--model", type=str, default='bert-base-uncased', help="the number of demonstration")

    return parser.parse_args()

class GLUETask(object):

    def __init__(self, task, train_file, valid_file, test_file):
        self.task = task

        if task in ['mr', 'cr', 'SST-2', 'sst-5', 'subj', 'trec', 'sst-2', 'ag_news', 'yelp_polarity', 'imdb']:
            self.train_inputs, self.train_labels = self.load_dataset(train_file, 0)
            self.valid_inputs, self.valid_labels = self.load_dataset(valid_file, 0)
            self.test_inputs, self.test_labels = self.load_dataset(test_file, 0)
        elif task in ['cb', 'mrpc', 'qnli', 'qqp', 'rte', 'snli']:
            self.train_inputs, self.train_labels = self.load_dataset(train_file, 1)
            self.valid_inputs, self.valid_labels = self.load_dataset(valid_file, 1)
            self.test_inputs, self.test_labels = self.load_dataset(test_file, 1)
        elif task == 'wic':
            self.train_inputs, self.train_labels = self.load_dataset(train_file, 2)
            self.valid_inputs, self.valid_labels = self.load_dataset(valid_file, 2)
            self.test_inputs, self.test_labels = self.load_dataset(test_file, 2)
        else:
            assert "error task name....."

    def load_dataset(self, file_name, data_type):  ### data_type: 0/1/2 -> single/pair/pair+word
        inputs = []
        labels = []
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                items = json.loads(line)
                if data_type == 0:
                    inputs.append(items['sentence'])
                elif data_type == 1:
                    if 'sentence1' in items:
                        inputs.append((items['sentence1'], items['sentence2']))
                    elif 'premise' in items:
                        inputs.append((items['premise'], items['hypothesis']))
                    elif 'question1' in items:
                        inputs.append((items['question1'], items['question2']))
                    else:
                        inputs.append((items['question'], items['sentence']))
                else:
                    inputs.append((items['sentence1'], items['sentence2'], items['word']))
                labels.append(int(items['label']))

        return inputs, labels

# # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
# def get_verbalizers_ids(task, tindex):
#     if task in ['mr', 'cr', 'SST-2', 'sst-2']:
#         return [" terrible", " great"]
#     elif task == 'sst-5':
#         return [" terrible", " bad", " okay", " good", " great"]
#     elif task == 'subj':
#         return [" subjective", " objective"]
#     elif task == 'trec':
#         return [" Description", " Entity", " Expression", " Human", " Location", " Number"]
#     elif task == 'rte':
#         if tindex == 4:
#             return [" true", " false"]
#         return [" Yes", " No"]
#     elif task == 'cb':
#         if tindex == 4:
#             return [' true', ' false', ' neither']
#         return [" Yes", " No", " Maybe"]
#     elif task == 'wic':
#         if tindex == 2:
#             return ["2", "b"]
#         return [" No", " Yes"]
#     elif task == 'qnli':
#         if tindex in [0, 2, 4]:
#             return [" Yes", " No"]
#         return [" true", " false"]
#     elif task in ['qqp', 'mrpc']:
#         if tindex in [0, 2, 4]:
#             return [" No", " Yes"]
#         return [" false", " true"]
#     else:
#         assert "error task name....."

# # we set verbalizers to the list of '<mask>' when using prompt
# def get_prompt_str(input_str, label_index, task, tindex, verbalizers): 
#     output_str = ''

#     if task in ['mr', 'cr', 'SST-2', 'sst-5', 'sst-2']:
#         if tindex == 0:
#             output_str = '{0} A{1} one.'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 1:
#             output_str = '{0} It was{1}.'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 2:
#             output_str = '{0} All in all{1}.'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 3:
#             output_str = '{0} A{1} piece.'.format(input_str, verbalizers[label_index]) 
#     elif task == 'subj':
#         if tindex == 0:
#             output_str = '{0} This is{1}.'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 1:
#             output_str = '{0} It\'s all{1}.'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 2:
#             output_str = '{0} It\'s{1}.'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 3:
#             output_str = '{0} Is it{1}?'.format(input_str, verbalizers[label_index]) 
#     elif task == 'trec':
#         if tindex == 0:
#             output_str = '{0}{1}:'.format(input_str, verbalizers[label_index]) 
#         elif tindex == 1:
#             output_str = '{0} Q:{1}:'.format(input_str, verbalizers[label_index])
#         elif tindex == 2:
#             output_str = '{0} why{1}?'.format(input_str, verbalizers[label_index])
#         elif tindex == 3:
#             output_str = '{0} Answer:{1}.'.format(input_str, verbalizers[label_index])
#     elif task in ['rte', 'cb']:   ### input: (premise, hypothesis)
#         if tindex == 0:
#             output_str = '"{1}"?{2}, "{0}"'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
#         elif tindex == 1:
#             output_str = '{1}?{2}, {0}'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
#         elif tindex == 2:
#             output_str = '"{1}"?{2}. "{0}"'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
#         elif tindex == 3:
#             output_str = '{1}?{2}. {0}'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
#         elif tindex == 4:
#             if task == 'rte':
#                 output_str = '{0} question: {1} True or False? answer:{2}.'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
#             else:
#                 output_str = '{0} question: {1} true, false or neither? answer:{2}.'.format(input_str[0], input_str[1].rstrip(string.punctuation), verbalizers[label_index])
#     elif task == 'wic':     ### input: (sentecne1, sentence2, word)
#         if tindex == 0:
#             output_str = '"{0}" / "{1}" Similar sense of "{2}"?{3}.'.format(input_str[0], input_str[1], input_str[2], verbalizers[label_index])
#         elif tindex == 1:
#             output_str = '{0} {1} Does {2} have the same meaning in both sentences?{3}.'.format(input_str[0], input_str[1], input_str[2], verbalizers[label_index])
#         elif tindex == 2:
#             output_str = '{2} . Sense (1) (a) "{0}" ({3}) "{1}"'.format(input_str[0], input_str[1], input_str[2], verbalizers[label_index])
#     elif task == 'qnli':    ### input: (question, sentence)
#         if tindex < 2:
#             output_str = '{1}. Question: {0}? Answer:{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
#         elif tindex < 4:
#             output_str = '{1}. Based on the previous sentence, {0}?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
#         else:
#             output_str = 'Based on the following sentence, {0}?{2}. {1}'.format(input_str[0], input_str[1], verbalizers[label_index])
#     elif task == 'qqp':   ### input: (question1, question2)
#         if tindex < 2:
#             output_str = 'Do "{0}" and "{1}" have the same meaning?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
#         elif tindex < 4:
#             output_str = '{0}. Based on the previous question, {1}?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
#         else:
#             output_str = 'Based on the following question, {0}?{2}. {1}'.format(input_str[0], input_str[1], verbalizers[label_index])
#     elif task == 'mrpc':  ### input: (sentence1, sentence2)
#         if tindex < 2:
#             output_str = 'Do "{0}" and "{1}" have the same meaning?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
#         elif tindex < 4:
#             output_str = '{0}. Based on the previous sentence, {1}?{2}.'.format(input_str[0], input_str[1], verbalizers[label_index])
#         else:
#             output_str = 'Based on the following sentence, {0}?{2}. {1}'.format(input_str[0], input_str[1], verbalizers[label_index])

#     return output_str

def get_basic_inputs(inputs, task, mask_token, sep_token):
    if task in ['mr', 'cr', 'SST-2', 'sst-5', 'subj', 'TREC', 'sst-2', 'ag_news', 'yelp_polarity', 'imdb']:
        new_inputs = [ input_str  for input_str in inputs ]
    elif task in ['CB', 'MRPC', 'QNLI', 'QQP', 'RTE', 'snli']:
        # new_inputs = [ input_str[0] + ' </s> ' + input_str[1] for input_str in inputs]  # concat trick for roberta
        new_inputs = [ input_str[0] + f' {sep_token} ' + input_str[1] for input_str in inputs]  # concat trick for bert
    elif task == 'WiC':
        new_inputs =[ input_str[0] + f' {sep_token} ' + input_str[1] + f' {sep_token} ' + input_str[2] for input_str in inputs]

    return new_inputs

def get_prompt_inputs(inputs, labels, task, tindex, verbalizers, mask_token, sep_token):
    new_inputs = []
    for input_str, label in zip(inputs, labels):
        new_input_str = get_prompt_str(input_str, label, task, tindex, verbalizers, mask_token, sep_token)
        new_inputs.append(new_input_str)

    return new_inputs

def get_prompt_demon_inputs(
        train_data_with_prompt, train_labels, inputs, labels, 
        task, tindex, num_label, num_demon, mask_token, sep_token,
        isSameData=False):
    none_verbalizers = [f' {mask_token}'] * num_label
    # get prompt-data for inputs
    new_prompt_inputs = get_prompt_inputs(inputs, labels, task, tindex, none_verbalizers, mask_token, sep_token)
    new_inputs = [new_prompt_inputs[index] + '\t' for index in range(len(new_prompt_inputs))]
    train_details = [(i, j) for i , j in zip(train_data_with_prompt, train_labels)]
    random.shuffle(train_details)
    train_data_with_prompt, train_labels = zip(*train_details)
    for cur_index in range(len(inputs)):
        used_ids = set()
        for sample_index in range(num_demon):
            total_sum = len(new_prompt_inputs[cur_index].split(' '))
            label_list = list(range(num_label))
            random.shuffle(label_list)
            for label_index in label_list:
                find_id = False
                for rid in range(len(train_labels)):
                    if train_labels[rid] == label_index:
                        if rid in used_ids:
                            continue
                        else:
                            demon_len = len(train_data_with_prompt[rid].split(' '))
                            if total_sum + demon_len < 512:
                                new_inputs[cur_index] += train_data_with_prompt[rid] + sep_token
                                total_sum += demon_len
                                used_ids.add(rid)
                                find_id = True
                                break

            if sample_index == num_demon - 1:
                new_inputs[cur_index] += new_prompt_inputs[cur_index] 
            else:
                new_inputs[cur_index] += new_prompt_inputs[cur_index] + '\t'

    return new_inputs

def save_data(file_name, inputs, labels):
    with open(file_name, 'w') as fw:
        for input_str, label in zip(inputs, labels):
            fw.write(input_str + '\t' + str(label) + '\n' )

def main(args):

    ### data processing
    if "roberta" in args.model:
        MASK_TOKEN = "<mask>"
        SEP_TOEKN = " </s>"
    elif "bert" in args.model:
        MASK_TOKEN = "[MASK]"
        SEP_TOEKN = " [SEP]"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    data = GLUETask(args.task, args.train, args.valid, args.test)
    new_train_labels = data.train_labels
    new_valid_labels = data.valid_labels
    new_test_labels = data.test_labels

    if args.mode == 0:
        new_train_inputs = get_basic_inputs(data.train_inputs, data.task, MASK_TOKEN, SEP_TOEKN)
        new_valid_inputs = get_basic_inputs(data.valid_inputs, data.task, MASK_TOKEN, SEP_TOEKN)
        new_test_inputs = get_basic_inputs(data.test_inputs, data.task, MASK_TOKEN, SEP_TOEKN)

        ### save data    
        save_data(args.output + '.train', new_train_inputs, new_train_labels)
        save_data(args.output + '.valid', new_valid_inputs, new_valid_labels)
        save_data(args.output + '.test', new_test_inputs, new_test_labels)

    elif args.mode == 1:
        none_verbalizers = [f' {MASK_TOKEN}'] * args.num_label
        
        for tindex in range(args.tindex):
            new_train_inputs = get_prompt_inputs(data.train_inputs, data.train_labels, data.task, tindex, none_verbalizers, MASK_TOKEN, SEP_TOEKN)
            new_valid_inputs = get_prompt_inputs(data.valid_inputs, data.valid_labels, data.task, tindex, none_verbalizers, MASK_TOKEN, SEP_TOEKN)
            new_test_inputs = get_prompt_inputs(data.test_inputs, data.test_labels, data.task, tindex, none_verbalizers, MASK_TOKEN, SEP_TOEKN)

            ### save data    
            save_data('{0}.tindex{1}.train'.format(args.output, tindex), new_train_inputs, new_train_labels)
            save_data('{0}.tindex{1}.valid'.format(args.output, tindex), new_valid_inputs, new_valid_labels)
            save_data('{0}.tindex{1}.test'.format(args.output, tindex), new_test_inputs, new_test_labels)

    elif args.mode == 2:
        random.seed(args.seed)

        for tindex in range(args.tindex):
            verbalizers = get_verbalizers_str(args.task, tindex)
            train_data_with_prompt = get_prompt_inputs(data.train_inputs, data.train_labels, data.task, tindex, verbalizers, MASK_TOKEN, SEP_TOEKN) # shot * num_labels

            new_train_inputs = get_prompt_demon_inputs(train_data_with_prompt, data.train_labels, \
                    data.train_inputs, data.train_labels, data.task, tindex, args.num_label, args.num_demon_train, MASK_TOKEN, SEP_TOEKN)
            new_valid_inputs = get_prompt_demon_inputs(train_data_with_prompt, data.train_labels, \
                    data.valid_inputs, data.valid_labels, data.task, tindex, args.num_label, args.num_demon_test, MASK_TOKEN, SEP_TOEKN)
            new_test_inputs = get_prompt_demon_inputs(train_data_with_prompt, data.train_labels, \
                    data.test_inputs, data.test_labels, data.task, tindex, args.num_label, args.num_demon_test, MASK_TOKEN, SEP_TOEKN)
                    
            ### save data    
            save_data('{0}.tindex{1}.train'.format(args.output, tindex), new_train_inputs, new_train_labels)
            save_data('{0}.tindex{1}.valid'.format(args.output, tindex), new_valid_inputs, new_valid_labels)
            save_data('{0}.tindex{1}.test'.format(args.output, tindex), new_test_inputs, new_test_labels)
            
    else:
        assert "error mode...."

if __name__ == '__main__':
    main(parse_args())
