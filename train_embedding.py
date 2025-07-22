import numpy as np
import random

# Create one hot vector
def one_hot(index, size):
    vec = np.zeros(size) #numpy lib, create array with all value = 0
    vec[index] = 1
    return vec

# Use in deep learning to random with rule:
# Each element is in the range (0, 1)
# The sum of all elements equals 1
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

### To use on target word to predict related context -> learn embedding vector that machine can understand
# embedding_dim: 10 -50 is lower value, for fast training, and simple data; higher is 100 -> 300+
# epoch: 10 - 100 lower training, 500 - 2000 able to learn better
# lr: is the step size used in each weight update; (0.0001–0.01) is low,  (0.05–1.0) is high, better when is small

def train_skipgram(pairs, vocab_size, embedding_dim=50, epochs=1000, lr=0.01):
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)

    for epoch in range(epochs):
        loss = 0
        random.shuffle(pairs)
        for target, context in pairs:
            x = one_hot(target, vocab_size)
            y_true = one_hot(context, vocab_size)
            
            h = np.dot(W1.T, x) # get hidden vector (embedding of target)
            u = np.dot(W2.T, h) # logits of each word
            y_pred = softmax(u) # random with softmax

            e = y_pred - y_true
            dW2 = np.outer(h, e) # Gradian for W2
            dW1 = np.outer(x, np.dot(W2, e)) # gradian for w1

            W1 -= lr * dW1
            W2 -= lr * dW2

            loss += -np.log(y_pred[context] + 1e-9)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1