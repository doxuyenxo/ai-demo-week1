from sklearn.metrics.pairwise import cosine_similarity


def recommend(query_idx, embeddings, documents):
    # To calculate similar from -1 to 1. Near 1 is more similar
    # cosine_sim(A,B) = (A * B) / (|A| * |B|)
    # eg: 
    # v0 = [0.8, 0.6]    # embedding of "I love machine learning" => 1
    # v1 = [0.78, 0.62]  # embedding of "I love AI" => 0.9998
    sims = cosine_similarity([embeddings[query_idx]], embeddings)[0]
    # sort by rank desc
    ranked = sims.argsort()[::-1]
    print("User is viewing:", documents[query_idx])
    print("\nRecommended:")
    for i in ranked[1:]:
        print(f"- ({sims[i]:.2f}) {documents[i]}")