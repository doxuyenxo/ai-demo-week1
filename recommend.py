from sklearn.metrics.pairwise import cosine_similarity


def recommend(query_idx, embeddings, documents):
    sims = cosine_similarity([embeddings[query_idx]], embeddings)[0]
    ranked = sims.argsort()[::-1]
    print("User is viewing:", documents[query_idx])
    print("\nRecommended:")
    for i in ranked[1:]:
        print(f"- ({sims[i]:.2f}) {documents[i]}")