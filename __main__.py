from tokenize_text import tokenize_text
from vocabulary import build_vocal
from skipgram_pairs import generate_skipgram_pairs
from train_embedding import train_skipgram
from embed_sentences import sentence_embedding
from recommend import recommend
import numpy as np

# Sample data
documents = [
    "Ultra-slim and lightweight laptop with high performance for work.",
    "Gaming laptop with powerful dedicated graphics card and high refresh rate display.",
    "Smartphone with dual cameras, sharp photo quality, and long battery life.",
    "Foldable smartphone with a unique design and premium look.",
    "Wireless computer mouse with ergonomic design to reduce hand fatigue.",
    "Mechanical keyboard with RGB LED lighting, offering a great typing experience."
]

# Tokenize documents
tokenized = [tokenize_text(doc) for doc in documents]
print("✅ Step 1 - Tokenized documents:")
for i, tokens in enumerate(tokenized):
    print(f"Doc {i+1}: {tokens}")

# Build word as index and index as word
word2idx, idx2word = build_vocal(tokenized)
print("\n✅ Step 2 - Vocabulary (word2idx):")
for word, idx in word2idx.items():
    print(f"{word}: {idx}")

# Generate skip-gram pairs;  Skip-Gram Word2Vec model use for training
pairs = generate_skipgram_pairs(tokenized, word2idx)
print("\n✅ Step 3 - Sample Skip-gram pairs:")
for i in range(len(pairs)):
    print(f"{idx2word[pairs[i][0]]} -> {idx2word[pairs[i][1]]}")

print("\n✅ Step 4 - Train Skip-gram:")
W1 = train_skipgram(pairs, vocab_size=len(word2idx))

# Embed each sentence. Sample:
# Doc 2 embedding[:5]: [0.12753465 0.40185069 0.42474644 0.70466724 0.18456377]
# Doc 2 embedding[:5]: [0.12753465 0.40185069 0.42474644 0.70466724 0.18456377]
# Doc 3 embedding[:5]: [0.45692182 0.18609687 0.42655762 0.3846459  0.25015412]
# Doc 4 embedding[:5]: [0.40543113 0.24724227 0.60482129 0.35293455 0.33394473]
# Doc 5 embedding[:5]: [0.42441382 0.25665996 0.4092651  0.30796706 0.47426697]
# Doc 6 embedding[:5]: [0.77976965 0.53588982 0.26711838 0.05712249 0.69555108]
# Doc 2 embedding[:5]: [0.12753465 0.40185069 0.42474644 0.70466724 0.18456377]
# Doc 3 embedding[:5]: [0.45692182 0.18609687 0.42655762 0.3846459  0.25015412]
# Doc 4 embedding[:5]: [0.40543113 0.24724227 0.60482129 0.35293455 0.33394473]
# Doc 5 embedding[:5]: [0.42441382 0.25665996 0.4092651  0.30796706 0.47426697]
# Doc 6 embedding[:5]: [0.77976965 0.53588982 0.26711838 0.05712249 0.69555108]
embeddings = [sentence_embedding(tokens, W1, word2idx, dim=W1.shape[1]) for tokens in tokenized]
print("\n✅ Step 5 - Sample sentence embeddings (first 5 dimensions):")
for i, emb in enumerate(embeddings):
    print(f"Doc {i+1} embedding[:5]: {emb[:5]}")

# Recommend based on first document
print("\n✅ Step 6 - Recommendations for first document:")
recommend(query_idx=0, embeddings=embeddings, documents=documents)