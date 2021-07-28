import argparse

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, BertModel


def clustering(opt):
    
    # prepare model
    print('[INFO] Prepare model')
    bert_config = BertConfig.from_json_file(
        '{}/config.json'.format(opt.model_path)
    )
    bert_config.output_hidden_states = True # 
    bert_tokenizer = BertTokenizer(
        '{}/vocab.txt'.format(opt.model_path),
        do_lower_case=False,
        do_basic_tokenize=False
    )
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    bert_model = BertModel.from_pretrained('{}/pytorch_model.bin'.format(opt.model_path), config=bert_config)
    #bert_model = BertForSequenceClassification.from_pretrained('{}/pytorch_model.bin'.format(opt.model_path), config=bert_config)
    bert_model.to(device)
    
    # load data
    print('[INFO] Prepare data')
    with open(opt.input_file) as input_file:
        documents = input_file.readlines()
    print('[INFO] Get document num {}'.format(len(documents)))
    
    # make tokenized sentence list 
    documents = [ document.strip() for document in documents ]
    
    # make tokens list using bert tokenizer
    documents = [ bert_tokenizer.tokenize(document) for document in documents ]
    
    # make words list <= 512 = ([CLS] + 510 + [SEP])
    #documents = [ document[:510] for document in documents ]
    documents = [ document[:126] for document in documents ]
    
    print('[INFO] Make representation from document using BERT model')
    vectors = []
    for document in tqdm(documents, ascii=True, dynamic_ncols=True):
        
        # covert token to ids
        ids = bert_tokenizer.convert_tokens_to_ids(document)
        ids = bert_tokenizer.build_inputs_with_special_tokens(ids)
        ids = torch.tensor([ids])
        ids = ids.to(device)
        
        # input model
        output = bert_model(ids)
        vector = output.last_hidden_state[0,0,:]
        #vector = output.hidden_states[12][0,0,:] # last_hidden_state
        #vector = output.logits[0] # output from classifier (low acc)
        vectors.append(vector.tolist())
        
    # vectors (document_num, hidden_size)
    np_vectors = np.array(vectors)
    
    if opt.cls_type == 'kmeans':
        # K-means clustering
        print('[INFO] K-means clustering')
        cls_model = KMeans(n_clusters=opt.num_cls, random_state=10).fit(np_vectors)
    else:
        # agglomerative clustering
        print('[INFO] Agglomerative clustering')
        cls_model = AgglomerativeClustering(
            n_clusters=opt.num_cls,
            distance_threshold=None,
            affinity=opt.cls_type,
            linkage='average',
        ).fit(np_vectors)
    
    # labels
    print('[INFO] Save labels as {}'.format(opt.output_file))
    labels = cls_model.labels_
    labels = labels.tolist()
    labels_str = '\n'.join(map(str, labels)) + '\n'
    with open(opt.output_file, mode='w') as f:
        f.write(labels_str)
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_type', type=str, choices=['kmeans', 'euclidean', 'cosine', 'manhattan'], default='kmeans')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_cls', type=int, default=2)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    opt = parser.parse_args()
    
    clustering(opt)
    
    
if __name__ == '__main__':
    main()

