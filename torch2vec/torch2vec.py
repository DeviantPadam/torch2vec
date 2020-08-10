#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:48:06 2020

@author: deviantpadam
"""

from torch2vec.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class NegativeSampling(nn.Module):
    
    
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, scores):
        
        k = scores.size()[1] - 1
        return -torch.sum(
            self._log_sigmoid(scores[:, 0])
            + torch.sum(self._log_sigmoid(-scores[:, 1:]), dim=1) / k
        ) / scores.size()[0]
    
    


class DM(nn.Module):
    """Distributed Memory version of Paragraph Vectors.
    Parameters
    ----------
    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).
    num_docs: int
        Number of documents in a dataset.
    num_words: int
        Number of distinct words in a daset (i.e. vocabulary size).
    """
    def __init__(self, vec_dim, num_docs, num_words):
        super(DM, self).__init__()
        # paragraph matrix
        self._D = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        # word matrix
        self._W = nn.Parameter(
            torch.randn(num_words, vec_dim), requires_grad=True)
        # output layer parameters
        self._O = nn.Parameter(
            torch.FloatTensor(vec_dim, num_words).zero_(), requires_grad=True)

    def forward(self, context_ids, doc_ids, target_noise_ids):
        
        
        # combine a paragraph vector with word vectors of
        # input (context) words
        x = torch.add(
            self._D[doc_ids, :], torch.sum(self._W[context_ids, :], dim=1))

        # sparse computation of scores (unnormalized log probabilities)
        # for negative sampling
        return torch.bmm(
            x.unsqueeze(1),
            self._O[:, target_noise_ids].permute(1, 0, 2)).squeeze()

    def get_paragraph_vector(self):
        return self._D.data.tolist()
    
    def fit(self,doc_ids,context,target_noise_ids,epochs,batch_size,
            num_workers=1):
        
        opt=torch.optim.Adam(self.parameters(),lr=0.0001)
        cost_func = NegativeSampling()
        if torch.cuda.is_available():            
            cost_func.cuda()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = Dataset(doc_ids, context, target_noise_ids)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                 num_workers=num_workers)
        loss = []
        for epoch in range(epochs):
            pbar = tqdm.tqdm(dataloader,
                        desc='Epoch= {} ---- prev loss={}'.format(epoch+1,loss))
            loss=[]
            
            for doc_ids,context_ids,target_noise_ids in pbar:
                doc_ids = doc_ids.to(device)
                context_ids = context_ids.to(device)
                target_noise_ids = target_noise_ids.to(device)
                x = self.forward(
                        context_ids,
                        doc_ids,
                        target_noise_ids) 
                x = cost_func.forward(x)
                loss.append(x.item())
                self.zero_grad()
                x.backward()
                opt.step()
#                 if step%100==0:
#                     print('-',end='')
            loss = torch.mean(torch.FloatTensor(loss))
#             print('epoch - {} loss - {:.4f}'.format(epoch+1,loss))
        print('Final loss: {:.4f}'.format(loss))
        
    def save_model(self,ids,file_name):
        docvecs = self._D.data.cpu().numpy()
        if len(docvecs)!=len(ids):
            raise Exception("Length of ids does'nt match")
            
            
        self.embeddings = np.concatenate([ids.reshape(-1,1),docvecs],axis=1)
        np.save(file_name,self.embeddings,fix_imports=False)
        
    def load_model(self,file_path):
        self.embeddings = np.load(file_path,allow_pickle=True,
                                  fix_imports=False)
        
    
    def similar_docs(self,docs,topk=10,use='torch'):
        topk=topk+1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not isinstance(docs,np.ndarray):
            docs = np.array(docs)
        
        docids = self.embeddings[:,0]
        vecs = self.embeddings[:,1:]
        mask = np.isin(docids,docs)
        if not mask.any():
            raise Exception('Not in vocab')
            
        given_docvecs = torch.FloatTensor(vecs[mask].tolist()).to(device)
        vecs = torch.FloatTensor(vecs.tolist()).to(device)
        similars= self._similarity(given_docvecs,vecs,topk,use=use)
        if use=='torch':
            similar_docs = docids[similars.indices.tolist()[0]].tolist()
            probs = similars.values.tolist()[0]
            return similar_docs[1:], probs[1:]
        if use=='sklearn':
            similar_docs = docids[similars].tolist()
#             probs = similars
            return similar_docs[1:]
        
    def _similarity(self,doc,embeddings,topk,use):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use=='torch':
            similarity = []
            cos=nn.CosineSimilarity(dim=0).to(device)
            for i in doc:
                inner = []
                pbar = tqdm.tqdm(embeddings,desc='----Getting Similar Docs----')
                for j in pbar:
                    inner.append(cos(i.view(-1,1),j.view(-1,1)).tolist())
                similarity.append(inner)
            similarity = torch.FloatTensor(similarity).view(1,-1).to(device)
            return torch.topk(similarity,topk), None
        if use=='sklearn':
            sim = cosine_similarity(X=doc.cpu().numpy(),
                                    Y=embeddings.cpu().numpy())
            index = np.flip(np.argsort(sim)[0][:topk])
            return index
    
    
class LoadModel():
    def __init__(self,path):
        self.embeddings = np.load(path,allow_pickle=True,fix_imports=False)
        
    def similar_docs(self,docs,topk=10,use='torch'):
        topk=topk+1
        if use not in ['torch','sklearn'] :
            raise Exception("Only 'sklearn' or 'torch' method can be used.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not isinstance(docs,np.ndarray):
            docs = np.array(docs)
        
        docids = self.embeddings[:,0]
        vecs = self.embeddings[:,1:]
        mask = np.isin(docids,docs)
        if not mask.any():
            raise Exception('Not in vocab')
            
        given_docvecs = torch.FloatTensor(vecs[mask].tolist()).to(device)
        vecs = torch.FloatTensor(vecs.tolist()).to(device)
        similars= self._similarity(given_docvecs,vecs,topk,use=use)
        if use=='torch':
            similar_docs = docids[similars.indices.tolist()[0]].tolist()
            probs = similars.values.tolist()[0]
            return similar_docs[1:], probs[1:]
        if use=='sklearn':
            similar_docs = docids[similars].tolist()
#             probs = similars
            return similar_docs[1:]
        
    def _similarity(self,doc,embeddings,topk,use):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use=='torch':
            similarity = []
            cos=nn.CosineSimilarity(dim=0).to(device)
            for i in doc:
                inner = []
                pbar = tqdm.tqdm(embeddings,desc='----Getting Similar Docs----')
                for j in pbar:
                    inner.append(cos(i.view(-1,1),j.view(-1,1)).tolist())
                similarity.append(inner)
            similarity = torch.FloatTensor(similarity).view(1,-1).to(device)
            return torch.topk(similarity,topk)
        if use=='sklearn':
            sim = cosine_similarity(X=doc.cpu().numpy(),
                                    Y=embeddings.cpu().numpy())
            index = np.flip(np.argsort(sim)[0][:topk])
            return index