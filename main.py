from tokenize_text import tokenize_text
from vocabulary import build_vocal
from skipgram_pairs import generate_skipgram_pairs
from train_embedding import train_skipgram
from embed_sentences import sentence_embedding
from recommend import recommend
import numpy as np

# Sample data
documents = [
    "Máy tính xách tay siêu mỏng nhẹ, hiệu năng cao cho công việc.",
    "Laptop gaming với card đồ họa rời mạnh mẽ, màn hình tần số quét cao.",
    "Điện thoại thông minh camera kép, chụp ảnh sắc nét, pin trâu.",
    "Smartphone màn hình gập độc đáo, thiết kế sang trọng.",
    "Chuột máy tính không dây, thiết kế công thái học, giảm mỏi tay.",
    "Bàn phím cơ có đèn LED RGB, trải nghiệm gõ phím tuyệt vời."
]

# Tokenize documents
tokenized = [tokenize_text(doc) for doc in documents]
print("Step 1 - Tokenized documents:")
for i, tokens in enumerate(tokenized):
    print(f"Doc {i+1}: {tokens}")

# Build word as index and index as word
word2idx, idx2word = build_vocal(tokenized)
print("\nStep 2 - Vocabulary (word2idx):")
for word, idx in word2idx.items():
    print(f"{word}: {idx}")

# Generate skip-gram pairs
pairs = generate_skipgram_pairs(tokenized, word2idx)
print("\nStep 3 - Sample Skip-gram pairs:")
for i in range(len(pairs)):
    print(f"{idx2word[pairs[i][0]]} -> {idx2word[pairs[i][1]]}")
W1 = train_skipgram(pairs, vocab_size=len(word2idx))

# Embed each sentence
embeddings = [sentence_embedding(tokens, W1, word2idx, dim=W1.shape[1]) for tokens in tokenized]
print("\nStep 5 - Sample sentence embeddings (first 5 dimensions):")
for i, emb in enumerate(embeddings):
    print(f"Doc {i+1} embedding[:5]: {emb[:5]}")

# Recommend based on first document
print("\nStep 6 - Recommendations for first document:")
recommend(query_idx=0, embeddings=embeddings, documents=documents)