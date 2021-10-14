import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA, PCA

import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/embeddings.pkl', type=str,
                        help="Path to the embeddings file.")
    parser.add_argument("-o", "--output", default='data/final/emb_{h}_{p}_{d}.pkl', type=str,
                        help="Path where to store the dimensionality reduced embeddings.")
    parser.add_argument("-p","--projection-method",default="tsne")
    parser.add_argument("-d","--dimension",default=2,
                        help="How many dimensions to reduce to.")
    parser.add_argument("-rh","--reduction-heuristic",default="cls")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
        
    output_name = args.output.format(
        h=args.reduction_heuristic,
        p=args.projection_method,
        d=args.dimension)
        
    # Read data
    with open(args.input,"rb") as f:
        embeddings = pickle.load(f)
    ids = embeddings["ids"]
    embeddings = embeddings["embeddings"]
    
    # Use reduction heuristic
    if args.reduction_heuristic == "cls":
        embeddings = embeddings[:,0]
    elif args.reduction_heuristic == "avg":
        embeddings = np.mean(embeddings,axis=1)
        
    if args.projection_method == "tsne":    
        reducer = TSNE(args.dimension,n_jobs=-1)
    elif args.projection_method == "pca":    
        reducer = PCA(args.dimension)
        
        
    projected_emb = reducer.fit_transform(embeddings)
    output = dict(ids=ids,
                  embeddings=projected_emb)
    
    with open(output_name,"wb") as f:
        pickle.dump(output,f)
    
    
    