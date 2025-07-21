import numpy as np

def sentence_embedding(tokens, W1, word2idx, dim):
    vectors = [W1[word2idx[word]] for word in tokens if word in word2idx]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)