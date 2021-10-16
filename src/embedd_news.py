import utils

from transformers import TFAutoModel,AutoTokenizer
import numpy as np

import pickle
import argparse
import operator as op
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/COP.clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/final/embeddings.pkl', type=str,
                        help="Path where to store the embeddings.")
    parser.add_argument("-t", "--target", default='headline', type=str,
                        help="Target field of the new to use for embedding.")
    parser.add_argument("-bs","--batch-size", default=128, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model to use for the embeddings.")
    parser.add_argument("--max-length", default=128, type=int,
                        help="Override the default maximum length of language model input.")
    args = parser.parse_args()
    return args

LM_ALIASES = dict(
    bert="bert-base-uncased",
    roberta="roberta-base",
    albert="albert-base-v2",
    distilroberta="distilroberta-base"
)

KEY_ID = "path"

if __name__ == "__main__":
    args = parse_args()
    
    # Read data
    data = utils.load_data(args.input)
    
    target = list(map(op.itemgetter(args.target),data))
    ids = list(map(op.itemgetter(KEY_ID),data))
    
    ## Instantiate transformer and tokenizer
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    lm = TFAutoModel.from_pretrained(lm_name)
    
    target = tokenizer(
            target, padding=True, max_length=args.max_length,
            truncation=True, return_tensors="np").data
    
    start_ids = tqdm.trange(0,len(data),args.batch_size)
    
    embeddings = []
    for s_i in start_ids:
        e_i = s_i + args.batch_size
        batch = {k: v[s_i:e_i] for k,v in target.items()}
        
        emb = lm(batch)[0].numpy()
        embeddings.append(emb)
        
    embeddings = np.concatenate(embeddings,axis=0)
    result = dict(
        ids=ids,
        embeddings=embeddings)

    with open(args.output,"wb") as f:
        pickle.dump(result,f)