# torch2vec
A PyTorch implementation of Doc2Vec (distributed memory) with similarity measure.

## Installation
### Dependencies
torch2vec requires:
* Python (>= 3.6)
* torch (>=1.6.0)
* numpy
* tqdm
* pandas
* scikit-learn

### User Installation
1. Clone the repository
<code> git clone https://github.com/DeviantPadam/torch2vec.git </code>
1. Go to repository directory
<code> cd torch2vec/ </code>
1. Run <code> pip install -U . </code>

## User Instructions 
### Data preprocessing
* Make sure your data is in the correct format as mentioned in **example_data/example.csv**.
* Import modules<br/> <code> from torch2vec.data import DataPreparation </code><br/><code>from torch2vec.torch2vec import DM </code>
* Now load your data for preprocessing.</br><code>data = DataPreparation(corpus_path='example_data/example.csv',vocab_size)</code> <br/> <code>vocab_size</code>: (optional)can be used to restrict vocabulary size (less frequent words will be dropped).
* Now create vocabulary using <code> data.vocab_builder() </code>
* Now get the doc_ids, context words, target words for further use<br/><code> doc, context, target_noise_ids = data.get_data(window_size,num_noise_words) </code><br/><code> window_size</code>: is the number of surrounding words. <br/> <code> num_noise_words</code>: is the number of words to be negative sampled.
### Training
* Initialize the model<br/> <code>model = DM(vec_dim=100,num_docs=len(data),num_words=data.vocab_size)</code><br/> <code>vec_dim</code>: Dimensions of documents vector<br/>
* Now train the model <br/><code>model.fit(doc_ids=doc,context=context,target_noise_ids=target_noise_ids,epochs=5,batch_size=1000,num_workers=2)</code><br/><code>doc_ids,context,target_noise_ids</code>: can be obtained using data.get_data<br/> <code>epochs</code>: number of epochs <br/> <code>batch_size</code>: batch size<br/> <code>num_workers</code>: (default=1) Number of concurrently running workers.(max=os.cpu_count())

* Now fit your real documents ids to doc embeddings and save the model(optional) <br/> <code>model.save_model(ids=data.document_ids,file_name='weights')</code> <br/> <code>file_name</code>: (optional) if None then model will not save.
* Now get similar document ids <br/> <code>model.similar_docs('doc_id',topk=10,use='torch')</code><br/> <code>topk</code>: (default=10) Get 'topk' numbers of similar docs <br/> <code>use</code>: 'torch' or 'sklearn' (deafault='torch') <br/> returns: similar ids and cosine similarity score of topk elements.(only similar ids if use='sklearn')  
* If model is saved (stored as .npy file) then model can be reused without training using <br/> <code>from torch2vec.torch2vec import LoadModel</code> <br/> <code> model = LoadModel(path='weights.npy')</code> <br/> Reusing: <code>model.similar_docs('doc_id',topk=10,use='torch')</code>

### References
* [Distributed Representations of Sentences and Documents Quoc V. Le, Tomas Mikolov](https://arxiv.org/pdf/1405.4053.pdf)
* [https://github.com/inejc/paragraph-vectors](https://github.com/inejc/paragraph-vectors)
* [Notes on Noise Contrastive Estimation and Negative Sampling, C. Dyer](https://arxiv.org/abs/1410.8251)
* [Document Embedding with Paragraph Vectors Andrew M. Dai, Christopher Olah, Quoc V. Le](https://arxiv.org/abs/1507.07998)

#### Special thanks to [Luc](https://github.com/x0rzkov) for helping and motivating me.
