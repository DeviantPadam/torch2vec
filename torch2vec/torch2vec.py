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
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = Dataset(doc_ids, context, target_noise_ids)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
        
        for epoch in range(epochs):
            pbar = tqdm.tqdm(dataloader,
                        desc='Epoch= {}'.format(epoch+1))
            loss=[]
            
            for doc_ids,context_ids,target_noise_ids in pbar:
                doc_ids = doc_ids.to(self.device)
                context_ids = context_ids.to(self.device)
                target_noise_ids = target_noise_ids.to(self.device)
                x = self.forward(
                        context_ids,
                        doc_ids,
                        target_noise_ids) 
                x = cost_func.forward(x)
                loss.append(x.item())
                pbar.set_postfix(loss='{:.4f}'.format(x.item()))
                self.zero_grad()
                x.backward()
                opt.step()
#                 if step%100==0:
#                     print('-',end='')
            loss = torch.mean(torch.FloatTensor(loss))
#             print('epoch - {} loss - {:.4f}'.format(epoch+1,loss))
#         print('Final loss: {:.4f}'.format(loss))
        
    def save_model(self,ids,args,file_name=None):
        docvecs = self._D.data.detach().cpu().numpy()
        if len(docvecs)!=len(ids):
            raise Exception("Length of ids does'nt match")
            
#         to_be_added = np.concatenate([arr.reshape(-1,1) for arr in args],axis=1)
        unk = np.unique(args.values)
        mapper = {word:idx for idx,word in enumerate(unk)}
        embed = nn.Embedding(len(docvecs)*args.shape[1],5).to(self.device)
        text = embed(torch.tensor([mapper[word] for idx in range(args.shape[1]) 
                                   for word in args.iloc[:,idx]],
                                  device=self.device)).to(self.device)
        text = text.detach().cpu().numpy()
        text = np.split(text,args.shape[1])
        to_be_added = np.concatenate(text,axis=1)
#         print([i.shape for i in text])
        self.embeddings = np.concatenate([ids.reshape(-1,1),to_be_added,docvecs]
                                         ,axis=1)
        if file_name:
            np.save(file_name,self.embeddings,fix_imports=False)
        
    def load_model(self,file_path):
        self.embeddings = np.load(file_path,allow_pickle=True,
                                  fix_imports=False)
        
    
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
            
        given_docvecs = torch.tensor(vecs[mask].tolist()).to(device)
        vecs = torch.tensor(vecs.tolist()).to(device)
        similars= self._similarity(given_docvecs,vecs,topk,use=use)
        if use=='torch':
            similar_docs = (docids[similars.indices.tolist()[0]]).tolist()
            probs = similars.values.tolist()[0]
            return similar_docs[1:], probs[1:]
        if use=='sklearn':
            similar_docs = docids[similars[1]].tolist()
#             probs = similars
            return similar_docs[1:], similars[0][1:]
        
    def _similarity(self,doc,embeddings,topk,use):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use=='torch':
            doc = doc/doc.norm(dim=1,keepdim=True)
            embeddings = embeddings/embeddings.norm(dim=1,keepdim=True)
            similarity = (torch.mm(doc,embeddings.transpose(0,1))).to(device)
            return torch.topk(similarity,topk)
        if use=='sklearn':
            sim = cosine_similarity(X=doc.detach().cpu().numpy(),
                                    Y=embeddings.detach().cpu().numpy())
            index = np.argsort(-sim)[0][:topk]
            sim = -np.sort(-sim[0])
            return sim[:topk].tolist(),index.tolist()
    
    
class LoadModel():
    def __init__(self,path,f_size=None):
        self.embeddings = np.load(path,allow_pickle=True,fix_imports=False)
        if f_size:
            self.f_size=f_size
        else:
            f_size=0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def similar_docs(self,docs,topk=10,use='torch'):
        topk=topk+1
        if use not in ['torch','sklearn'] :
            raise Exception("Only 'sklearn' or 'torch' method can be used.")
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not isinstance(docs,np.ndarray):
            docs = np.array(docs)
        
        docids = self.embeddings[:,0]
        vecs = self.embeddings[:,self.f_size*5+1:]
        mask = np.isin(docids,docs)
        if not mask.any():
            raise Exception('Not in vocab')
          
        given_docvecs = torch.tensor(vecs[mask].tolist(),device=self.device) 
        vecs = torch.tensor(vecs.tolist(),device=self.device)
        if self.f_size:
            args = self.embeddings[:,1:self.f_size+1]
            args = torch.tensor(args.tolist(),device=self.device)
            given_args = torch.tensor(args[mask].tolist(),device=self.device)
            similars= self._similarity_args(given_docvecs,vecs,args,given_args,
                                            topk,use=use)
        else:
            similars= self._similarity(given_docvecs,vecs,topk,use=use)
        if use=='torch':
            similar_docs = (docids[similars.indices.tolist()[0]]).tolist()
            probs = similars.values.tolist()[0]
            return similar_docs[1:], probs[1:]
        if use=='sklearn':
            similar_docs = docids[similars[1]].tolist()
#             probs = similars
            return similar_docs[1:], similars[0][1:]
        
    def _similarity(self,doc,embeddings,topk,use):
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if use=='torch':
            doc = doc/doc.norm(dim=1,keepdim=True)
            embeddings = embeddings/embeddings.norm(dim=1,keepdim=True)
            similarity = (torch.mm(doc,embeddings.transpose(0,1))).to(self.device)
            return torch.topk(similarity,topk)
        if use=='sklearn':
            sim = cosine_similarity(X=doc.detach().cpu().numpy(),
                                    Y=embeddings.detach().cpu().numpy())
            index = np.argsort(-sim)[0][:topk]
            sim = -np.sort(-sim[0])
            return sim[:topk].tolist(),index.tolist()
    
    def _similarity_args(self,doc,embeddings,args,given_args,topk,use):
        
        
        if use=='torch':
            doc = doc/doc.norm(dim=1,keepdim=True)
            embeddings = embeddings/embeddings.norm(dim=1,keepdim=True)
            similarity1 = (torch.mm(doc,embeddings.transpose(0,1))).to(self.device)
            args = args/args.norm(dim=1,keepdim=True)
            given_args = given_args/given_args.norm(dim=1,keepdim=True)
            similarity2 = (torch.mm(given_args,args.transpose(0,1))).to(self.device)
            total = (similarity1+similarity2)/2
            return torch.topk(total,topk)
        if use=='sklearn':
            sim = cosine_similarity(X=doc.detach().cpu().numpy(),
                                    Y=embeddings.detach().cpu().numpy())
            index = np.argsort(-sim)[0][:topk]
            sim = -np.sort(-sim[0])
            return sim[:topk].tolist(),index.tolist()