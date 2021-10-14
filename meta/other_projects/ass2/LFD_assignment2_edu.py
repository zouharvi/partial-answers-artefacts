import argparse
from argparse import Namespace

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report


def parse_args() -> Namespace:
    """Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-file", default='reviews.txt', type=str, required=True,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-ts", "--test-set", type=str, required=True,
                        help="Input file to test system")
    
    args = parser.parse_args()
    return args
    
def read_corpus(corpus_filepath: str) -> tuple[list[list[str]], list[str]]:
    """Read and parse the corpus from file.

    Parameters
    ==========
        - "corpus_filepath": filepath of the file to be read.

    Returns
    =======
        A 2-tuple containing:
            1. The tokenized sentences.
            2. The opic labels for each respective sentence.
    """

    documents = []
    labels_s = []
    labels_m = []
    with open(corpus_filepath, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(" ".join(tokens[3:]))
            
            # 6-class problem: books, camera, dvd, health, music, software
            labels_m.append(tokens[0])

    return documents, labels_m
    
if __name__ == "__main__":
    args = parse_args()
    
    # Read data
    X_train, Y_train = read_corpus(args.input_file)
    X_test, Y_test = read_corpus(args.test_set)
    
    # Convert to numpy
    X_train, Y_train, X_test, Y_test = map(np.array,(X_train,Y_train,X_test,Y_test))
    
    # Instantiate vectorizer
    tfidf =TfidfVectorizer(
        ngram_range=(1,2),
        max_df=0.85,
        lowercase=True,
        max_features=100000)
 
    # Vectorize input
    tfidf.fit(X_train)
    X_train, X_test = map(tfidf.transform,(X_train,X_test))
    
    # Instantiate model
    NB = GridSearchCV(ComplementNB(),param_grid=dict(),cv=10,n_jobs=-1)
    
    # Train model
    NB.fit(X_train,Y_train)
    
    # Test model
    Y_pred = NB.predict(X_test)
    Y_proba = NB.predict_proba(X_test)
    print(classification_report(Y_test,Y_pred))
    
        
    