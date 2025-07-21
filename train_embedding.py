import numpy as np
import random

def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def train_skipgram(pairs, vocab_size, embedding_dim=50, epochs=1000, lr=0.01):
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)

    for epoch in range(epochs):
        loss = 0
        random.shuffle(pairs)
        for target, context in pairs:
            x = one_hot(target, vocab_size)
            y_true = one_hot(context, vocab_size)
            
            h = np.dot(W1.T, x)
            u = np.dot(W2.T, h)
            y_pred = softmax(u)

            e = y_pred - y_true
            dW2 = np.outer(h, e)
            dW1 = np.outer(x, np.dot(W2, e))

            W1 -= lr * dW1
            W2 -= lr * dW2

            loss += -np.log(y_pred[context] + 1e-9)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1