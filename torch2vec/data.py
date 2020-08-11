#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:48:57 2020

@author: deviantpadam(Padam Gupta)
"""
import re
import tqdm
import numpy as np
import pandas as pd
import torch
from numpy.random import choice
from collections import Counter


class DataPreparation():
    def __init__(self,corpus_file_path,vocab_size=None):
        if '.txt' in corpus_file_path: 
            data = pd.read_csv(corpus_file_path,delimiter='\t')
        if '.csv' in corpus_file_path:
            data = pd.read_csv(corpus_file_path)
        self.corpus = data.iloc[:,1]
        self.document_ids = data.iloc[:,0].values
#         self.window_size = window_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab_size = vocab_size if vocab_size else None
        self.stopwords = ['one', 'very', 'behind', 'move', 'yours', 'through', 
                          '‘ve', 'much',
       'your', 'just', '’s', 'call', 'therein', "n't", 'nor', 'almost',
       'keep', 'my', 'since', 'each', 'rather', 'two', 'did', 'put',
       'anyone', 'less', 'made', 'against', 'i', 'herself', 'why',
       'unless', 'her', 'third', 'many', '‘re', 'besides', 'anything',
       'whereby', 'already', 'his', 'our', 'may', 'see', 'then', 'as',
       'twenty', 'off', '‘d', 'twelve', 'do', 'could', 'have', 'the',
       'up', 'some', 'across', 'a', "'s", 'everyone', 'am', 'serious',
       'are', 'now', 'upon', 'is', 'make', "'d", 'really', 'might',
       'give', '‘s', 'us', 'but', 'whose', 'him', 'sometime', 'nine',
       'of', 'n’t', 'which', 'becomes', 'three', 'himself', 'any',
       'while', 'been', 'hers', 'still', 'else', 'ourselves', 'six',
       'has', '’ll', 'alone', 'fifteen', 'hence', 'after', 'four', 'most',
       'ours', 'few', 'several', 'beyond', 'about', '’ve', 'from', 'thru',
       'seems', '‘ll', 'onto', 'thus', 'empty', 'mine', 'whole', 'name',
       'was', 'latterly', 'or', 'when', 'became', 'these', 'would', 'be',
       'such', 'so', 'get', 'beforehand', 'wherever', 'formerly',
       'though', 'namely', 'own', "'re", 'nothing', 'please', 'sometimes',
       'before', 'because', 'and', 'during', 'various', 'along', 'cannot',
       '’re', 'around', 'must', 'regarding', 'become', 'yet', 'whoever',
       '’d', 'eight', 'doing', 'their', 'herein', 'without', 'eleven',
       'noone', 'moreover', 'others', 'part', 'least', 'again', 'top',
       'should', 'its', 'thereby', 'where', 'indeed', 'whether', 'out',
       "'ll", 'how', 'sixty', 'anyway', 'take', 'whence', "'ve", 'if',
       'perhaps', 'everything', 'had', 'even', 'whereas', "'m", 'all',
       'hereafter', 'at', 'no', 'anywhere', 'once', '’m', 'often', 'done',
       'other', 'hundred', 'go', 'whereafter', 'for', 'he', 'below',
       'she', 'enough', 'except', 'an', 'whereupon', 'somehow', 'it',
       'afterwards', 'toward', 'side', 'becoming', 'anyhow', 'further',
       'every', 'we', 'first', 'used', 'were', 'ca', 'something', 'those',
       'thence', 'due', 'itself', 'say', 'whatever', 'hereupon',
       'nowhere', 'this', 'to', 'on', 'among', 'both', 'either', 'being',
       'always', 'neither', 'show', 'thereafter', 'whither', 'over',
       'full', 'fifty', 'nobody', 'between', 'by', 'will', 'seemed',
       'another', 'only', 'per', 'under', 'someone', 'until', 'meanwhile',
       'into', 'also', 'ten', 'everywhere', 'well', 'what', 'latter',
       'in', 'towards', 'more', 'too', 'throughout', 'they', 'yourselves',
       'next', 'seem', 'beside', 'ever', 'former', 'n‘t', 'elsewhere',
       'that', 're', 'myself', 'yourself', 'whenever', 'via', 'amongst',
       'not', 'therefore', 'front', 'although', 'above', 'themselves',
       'somewhere', 'than', 'whom', 'together', 'using', '‘m', 'wherein',
       'does', 'hereby', 'them', 'who', 'five', 'seeming', 'however',
       'same', 'with', 'you', 'nevertheless', 'forty', 'down', 'quite',
       'can', 'none', 'here', 'otherwise', 'never', 'back', 'within',
       'mostly', 'amount', 'last', 'bottom', 'me', 'there', 'thereupon',
       '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-',
       '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^',
       '_', '`', '{', '|', '}', '~', '-pron-', '-PRON-']
        
    def vocab_builder(self):
        tqdm.tqdm.pandas(desc='--- Tokenizing ---')
        self.corpus = self.corpus.progress_apply(self._tokenize_str)
        vocab = [word for sentence in self.corpus.values for word in sentence]
        word_counts = Counter(vocab)
        if not self.vocab_size:
            self.vocab_size = len(vocab)
        self.word_counts = word_counts.most_common()[:self.vocab_size]
        self.vocab = [word[0] for word in self.word_counts]+['[UNK]']
        self.vocab_size = len(self.vocab)
        self.word_id_mapper = {word:ids for ids,word in enumerate(self.vocab)}
        self.id_word_mapper = dict(zip(self.word_id_mapper.values(),
                                       self.word_id_mapper.keys()))
            
    
    def _tokenize_str(self,str_):
        
        # keep only alphanumeric and punctations
        str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
        # remove multiple whitespace characters
        str_ = re.sub(r'\s{2,}', ' ', str_)
        # punctations to tokens
        str_ = re.sub(r'\(', ' ( ', str_)
        str_ = re.sub(r'\)', ' ) ', str_)
        str_ = re.sub(r',', ' , ', str_)
        str_ = re.sub(r'\.', ' . ', str_)
        str_ = re.sub(r'!', ' ! ', str_)
        str_ = re.sub(r'\?', ' ? ', str_)
        # split contractions into multiple tokens
        str_ = re.sub(r'\'s', ' \'s', str_)
        str_ = re.sub(r'\'ve', ' \'ve', str_)
        str_ = re.sub(r'n\'t', ' n\'t', str_)
        str_ = re.sub(r'\'re', ' \'re', str_)
        str_ = re.sub(r'\'d', ' \'d', str_)
        str_ = re.sub(r'\'ll', ' \'ll', str_)
        # lower case

        return [word for word in str_.strip().lower().split() 
                if word not in self.stopwords and len(word)>2]
    
    def get_data(self,window_size,num_noise_words):
        '''
        num_noise_words: number of words to be negative sampled
        '''
        self._padder(window_size)
        data = self._corpus_to_num()
        instances = self._instance_count(window_size)
        context = np.zeros((instances,window_size*2+1),dtype=np.int32)
        doc = np.zeros((instances,1),dtype=np.int32)
        k = 0 
        for doc_id, sentence  in (enumerate(tqdm.tqdm(data,
                                            desc='---- Creating Data ----'))):
            for i in range(window_size, len(sentence)-window_size):
                context[k] = sentence[i-window_size:i+window_size+1]
                doc[k] = doc_id
                k += 1
                
        target = context[:,window_size]
        context = np.delete(context,window_size,1)
        doc = doc.reshape(-1,)
        target_noise_ids = self._sample_noise_distribution(num_noise_words,
                                                           window_size)
        target_noise_ids = np.insert(target_noise_ids,0,target,axis=1)
        
        
        context = torch.from_numpy(context).type(torch.LongTensor)
        doc = torch.from_numpy(doc).type(torch.LongTensor)
        target_noise_ids = torch.from_numpy(target_noise_ids).type(torch.LongTensor)
        

        
        return doc,context,target_noise_ids
            
    def _padder(self,window_size):
        for i in range(len(self.corpus.values)):
            self.corpus.values[i] = ('[UNK] '*window_size).strip().split()+\
                self.corpus.values[i]+('[UNK] '*window_size).strip().split()
            
    def _corpus_to_num(self):
        num_corpus = []
        unk_count = 0
        for sentence in self.corpus.values:
            sen = []
            for word in sentence:
                if word in self.word_id_mapper:
                    sen.append(self.word_id_mapper[word])
                else:
                    sen.append(self.word_id_mapper['[UNK]'])
                    unk_count+=1
            num_corpus.append(sen)
            
        self.word_counts+=[('[UNK]',unk_count)]
        return np.array(num_corpus,dtype='object')
    
    def _instance_count(self,window_size):
        instances = 0
        for i in self.corpus.values:
            instances+=len(i)-2*window_size   
        return instances
        
    def _sample_noise_distribution(self,num_noise_words,window_size):
        
        probs = np.zeros(self.vocab_size)

        for word, freq in self.word_counts:
            probs[self.word_id_mapper[word]] = freq

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        return choice(probs.shape[0],(self._instance_count(window_size),
                                      num_noise_words),
                      p=probs).astype(np.int32)
    
    def __len__(self):
        return len(self.corpus)
    
    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self,doc_ids,context, target_noise_ids):
        self.doc_ids = doc_ids
        self.context = context
        self.target_noise_ids = target_noise_ids
        
    def __len__(self):
        return len(self.doc_ids)
    
    def __getitem__(self,index):
        return self.doc_ids[index], self.context[index], self.target_noise_ids[index]
