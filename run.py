#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:20:11 2020

@author: deviantpadam
"""

import os
import torch
from torch2vec.torch2vec import DM
from torch2vec.data import DataPreparation
# pd.read_csv('../input/')
num_workers = os.cpu_count()
def start(data_path,window_size,num_noise_words,epochs,batch_size,file_name,vec_dim=100,vocab_size=None,num_workers=num_workers):

    data = DataPreparation(data_path,vocab_size) #vocab_size to restrict vocabulary size
    
    data.vocab_builder()
    
    doc, context, target_noise_ids = data.get_data(window_size=window_size,
                                                   num_noise_words=num_noise_words)
    model = DM(vec_dim=vec_dim,num_docs=len(data),num_words=data.vocab_size)
    if torch.cuda.is_available():
        model.cuda()
    
    
    model.fit(doc_ids=doc,context=context,target_noise_ids=target_noise_ids,epochs=epochs,batch_size=batch_size,num_workers=num_workers)
    
    model.save_model(ids=data.document_ids,file_name='weights')
