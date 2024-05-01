import faiss
import json
import numpy as np
import os
import torch
from transformers import AutoModel, AutoTokenizer

def create_faiss_index():

    print('\nLoading embedder. Please wait . . .')

    embedder_name = 'BAAI/bge-small-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(embedder_name)
    embedder = AutoModel.from_pretrained(embedder_name)
        
    embeddings = []
    element_ids = []

    print('\nCreating embeddings. Please wait . . .')

    for filename in os.listdir('./custom_knowledge_documents'):

        if filename.endswith('.json'):

            with open(os.path.join('./custom_knowledge_documents/', filename), 'r') as file:

                elements = json.load(file)

                for element in elements:

                    element_id = element.get('element_id')

                    text = element.get('text')

                inputs = tokenizer(text, return_tensors = 'pt', padding = True, truncation = True)

                with torch.no_grad():

                    output = embedder(**inputs).last_hidden_state.mean(dim = 1)

                    embeddings.append(output.numpy())
                    
                    element_ids.append(element_id)      

    embeddings = np.concatenate(embeddings, axis = 0)

    # Dimension of vectors (Must be a multiple of "m")
    d = 384
    # Number of clusters
    nlist = 1
    # Number of sub-quantizers
    m = 8

    print('\nCreating vector database. Please wait . . .')

    quantizer = faiss.IndexFlatL2(d)

    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # "8" specifies that each sub-vector is encoded as 8 bits
    #index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    assert not index.is_trained

    index.train(embeddings)
    
    assert index.is_trained

    index.add(embeddings)
    
    faiss.write_index(index, os.path.join('./vector_database', 'faiss_index.index'))

    print('\nDone! Vector database created!')

    return(index)                

if __name__ == '__main__':

    faiss_index = create_faiss_index()        
