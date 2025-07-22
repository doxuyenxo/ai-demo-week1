from collections import defaultdict

# tokenized_documents = [
#     ["hello", "world"],
#     ["hello", "machine", "learning"]
# ]

# =>  output
# {
#     "hello": 0,
#     "world": 1,
#     "machine": 2,
#     "learning": 3
# }

def build_vocal(tokenized_documents):
    word_counts = defaultdict(int)
    for doc in tokenized_documents:
        for word in doc:
            word_counts[word] += 1
    word2idx = {word: idx for idx, word in enumerate(word_counts.keys())}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word