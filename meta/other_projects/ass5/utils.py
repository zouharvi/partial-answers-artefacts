import json
import numpy as np
from sklearn.metrics import accuracy_score


def report_accuracy_score(Y_dev, Y_pred):
    Y_dev = np.argmax(Y_dev, axis=1)
    Y_pred = np.argmax(Y_pred, axis=1)
    print(f"ACC: {accuracy_score(Y_dev, Y_pred):.3%}")


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def center_norm(X_data):
    X_data -= X_data.mean(axis=0)
    X_data /= np.linalg.norm(X_data, axis=0)

    # print(np.linalg.norm(reviews_all[0,:]))
    # print(np.linalg.norm(reviews_all[:,0]))

    return X_data
