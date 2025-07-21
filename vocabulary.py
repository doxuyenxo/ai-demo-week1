from collections import defaultdict

def build_vocal(tokenized_documents):
    word_counts = defaultdict(int)
    for doc in tokenized_documents:
        for word in doc:
            word_counts[word] += 1
    word2idx = {word: idx for idx, word in enumerate(word_counts.keys())}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word