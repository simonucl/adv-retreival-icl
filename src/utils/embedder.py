from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Union
import torch
# from pyserini.search import SimpleSearcher
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from InstructorEmbedding import INSTRUCTOR
from tqdm import tqdm
import multiprocessing
import os
import pickle
from torch.multiprocessing import Pool, Process, set_start_method

instructor_suffix = (' for retrieval: ', ' for retrieving support documents: ')

instructor_prefix = {
    'sst2': 'Represent the sentence',
    'rte': 'Represent the document',
    'mnli': 'Represent the document',
    'mr': 'Represent the sentence',
    'cr': 'Represent the sentence',
    'trec': 'Represent the sentence',
}


class IndexEmbedder(torch.nn.Module):
    def __init__(self, model_name, task_name, retrieval_method='sbert'):
        super().__init__()
        self.retrieval_method = retrieval_method

        if retrieval_method == 'sbert':
            self.embedder = SentenceTransformer(model_name)
        elif retrieval_method == 'instructor':
            self.instructor = INSTRUCTOR('hkunlp/instructor-large')
        elif retrieval_method == 'bm25':
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError(f'Retrieval method {retrieval_method} not implemented')
        self.task_name = task_name
        self.cache = None

    def encode(self, queries: List[str]) -> torch.Tensor:
        return self.embedder.encode(queries, convert_to_tensor=True)

    def bm25subsample(self, 
                      anchor_data : List[str], 
                      original_anchor, 
                      query : List[str],
                      top_k=1, 
                      num_labels=2) -> List[List[str]]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''

        retrieved_examples = [[] for _ in range(len(query))]

        if self.cache is None:
            tokenized_corpus = [self.bert_tokenizer.tokenize(doc) for doc in anchor_data]
            bm25 = BM25Okapi(tokenized_corpus)
        else:
            bm25 = self.cache

        for i, q in enumerate(query):
            tokenized_query = self.bert_tokenizer.tokenize(q)
            doc_scores = bm25.get_scores(tokenized_query)
            top_results = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i])[-top_k*num_labels:]
            for idx in top_results:
                retrieved_examples[i].append(original_anchor[idx])

        return retrieved_examples
    
    def sbert_subsample(self, 
                        anchor_data : List[str], 
                        original_anchor, 
                        query : List[str], 
                        top_k=1, 
                        num_labels=2,
                        batch_size=128,
                        ) -> List[List[str]]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''
        retrieved_examples = [[] for _ in range(len(query))]

        if self.cache is None:
            anchor_data_embeddings = self.embedder.encode(anchor_data, convert_to_tensor=True, batch_size=batch_size)
        else:
            anchor_data_embeddings = self.cache
        
        query_embedding = self.embedder.encode(query, convert_to_tensor=True, batch_size=batch_size)

        print(f'anchor_data_embeddings shape: {anchor_data_embeddings.shape}')
        print(f'query_embedding shape: {query_embedding.shape}')

        cos_scores = util.pytorch_cos_sim(query_embedding, anchor_data_embeddings)
        print(f'cos_scores shape: {cos_scores.shape}')
        top_results = torch.topk(cos_scores, k=top_k*num_labels, dim=1)
        print(f'top_results shape: {top_results[0].shape}, {top_results[1].shape}')
        for i in range(top_results[0].shape[0]):
            for j in range(top_results[0].shape[1]):
                retrieved_examples[i].append(original_anchor[top_results[1][i][j].item()])
        # for score, idx in zip(top_results[0], top_results[1]):
        #     for i, (s, id) in enumerate(zip(score, idx)):
        #         retrieved_examples[i].append(original_anchor[id.item()])
        return retrieved_examples

    def instructor_subsample(self, 
                             anchor_data : List[str],
                                original_anchor,
                                query : List[str],
                                top_k=1,
                                num_labels=2,
                                batch_size=128,
                             ) -> List[List[str]]:
        '''
        queries: list of query, Shape: [B, seq_len]
        corpus: list of corpus, Shape: [B, seq_len]
        returns: top-k retrieved corpus
        '''
        retrieved_examples = [[] for _ in range(len(query))]

        if self.cache is None:
            anchor_data = [[instructor_prefix[self.task_name] + instructor_suffix[0], doc] for doc in anchor_data]
            anchor_data_embeddings = self.instructor.encode(anchor_data, batch_size=batch_size, convert_to_tensor=True)
        else:
            anchor_data_embeddings = self.cache

        print(f'anchor_data_embeddings shape: {anchor_data_embeddings.shape}')
        
        query = [[instructor_prefix[self.task_name] + instructor_suffix[1], q] for q in query] # [B, 2]
        print(f'query shape: {len(query)}')
        query_embedding = self.instructor.encode(query, batch_size=batch_size, convert_to_tensor=True) # [B, 768]
        print(f'query_embedding shape: {query_embedding.shape}')

        cos_scores = util.pytorch_cos_sim(query_embedding, anchor_data_embeddings) # [B, len(anchor_data)]
        print(f'cos_scores shape: {cos_scores.shape}')

        top_results = torch.topk(cos_scores, k=top_k*num_labels, dim=1) # [B, top_k*num_labels], [B, top_k*num_labels]
        for i in range(top_results[0].shape[0]):
            for j in range(top_results[0].shape[1]):
                retrieved_examples[i].append(original_anchor[top_results[1][i][j].item()])
        return retrieved_examples

    def process_text(self, i, text, anchor_data_idx, original_anchor_data, retrieve_method, top_k, num_labels):
        if retrieve_method == 'sbert':
            result = self.sbert_subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        elif retrieve_method == 'bm25':
            result = self.bm25subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        elif retrieve_method == 'instructor':
            result = self.instructor_subsample(anchor_data_idx, original_anchor_data, text, top_k, num_labels)
        else:
            raise NotImplementedError(f'Retrieval method {retrieve_method} not implemented')
        return i, result
    
    def encode_anchor_data(self, anchor_data, batch_size=128):
        if type(anchor_data[0]) is dict:
            anchor_data = list(map(lambda x: x['sentence'] if 'sentence' in x else x['hypothesis'], anchor_data))
        elif type(anchor_data[0]) is tuple:
            anchor_data = list(map(lambda x: x[0], anchor_data))

        if self.retrieval_method == 'sbert':
            return self.embedder.encode(anchor_data, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
        elif self.retrieval_method == 'instructor':
            anchor_data = [[instructor_prefix[self.task_name] + instructor_suffix[0], doc] for doc in anchor_data]
            return self.instructor.encode(anchor_data, show_progress_bar=True, batch_size=batch_size, convert_to_tensor=True)
        elif self.retrieval_method == 'bm25':
            tokenized_corpus = [self.bert_tokenizer.tokenize(doc) for doc in tqdm(anchor_data, desc='Tokenizing anchor data')]
            bm25 = BM25Okapi(tokenized_corpus)
            return bm25
        else:
            raise NotImplementedError(f'Retrieval method {self.retrieval_method} not implemented')

    
    def subsamplebyretrieval(self, 
                             anchor_data : Union[List[Dict], List[List[Dict]]], 
                             text_input_list, 
                             top_k=1, 
                            num_labels=2,
                            retrieve_method='sbert',
                            save_path=None) -> List[List[str]]:
        '''
        anchor_data: list of anchor data, [{'sentence': 'text', 'label': 0}, ...]
        text_input_list: list of input text, Shape: [B, seq_len]
        returns: top-k retrieved anchor data
        '''
        # print(len(anchor_data))
        retrieved_examples = None
        if save_path is None or not os.path.exists(save_path):

            if type(anchor_data[0]) is not list:
                self.cache = self.encode_anchor_data(anchor_data)
                print(f'Finished encoding anchor data')

                text_input_list = [text[0] if type(text) is tuple else text for text in text_input_list]

                processed_text = self.process_text(0, text_input_list, anchor_data, anchor_data, retrieve_method, top_k*2, num_labels)
                retrieved_examples = processed_text[1]
            else:
                assert (len(anchor_data) == len(text_input_list)), f'Length of anchor data {len(anchor_data)} and text input list {len(text_input_list)} must be the same'
                retrieved_examples = [[] for _ in range(len(text_input_list))]

                for i, text in enumerate(tqdm(text_input_list, desc='Retrieving anchor data')):
                    if type(text) is tuple:
                        text = text[0]
                    if type(anchor_data[0]) is list:
                        anchor_data_idx = anchor_data[i]
                        original_anchor_data = anchor_data[i]
                    else:
                        anchor_data_idx = anchor_data
                        original_anchor_data = anchor_data

                    if type(anchor_data_idx[0]) is dict:
                        anchor_data_idx = list(map(lambda x: x['sentence'] if 'sentence' in x else x['hypothesis'], anchor_data_idx))
                    elif type(anchor_data_idx[0]) is tuple:
                        anchor_data_idx = list(map(lambda x: x[0], anchor_data_idx))

                    processed_text = self.process_text(i, text, anchor_data_idx, original_anchor_data, retrieve_method, top_k*2, num_labels)
                    retrieved_examples[i] = processed_text[1]

            with open(save_path, 'wb') as f:
                print(f'Saving retrieved examples to {save_path}')
                pickle.dump(retrieved_examples, f)

        if retrieved_examples is None:
            with open(save_path, 'rb') as f:
                print(f'Loading retrieved examples from {save_path}')
                retrieved_examples = pickle.load(f)
        assert len(retrieved_examples) == len(text_input_list), f'Length of retrieved examples {len(retrieved_examples)} and text input list {len(text_input_list)} must be the same'
        
        return [retrieved_example[:top_k*num_labels] for retrieved_example in retrieved_examples]