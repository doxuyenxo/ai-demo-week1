def generate_skipgram_pairs(tokenized_docs, word2idx, window_size=2):
    pairs = []
    for doc in tokenized_docs:
        for i, target_word in enumerate(doc):
            for j in range(max(0, i - window_size), min(len(doc), i + window_size + 1)):
                if i != j:
                    pairs.append((word2idx[target_word], word2idx[doc[j]]))
    return pairs