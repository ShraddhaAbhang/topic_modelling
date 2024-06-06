from sklearn.decomposition import LatentDirichletAllocation

def perform_lda(vectorized_texts, n_topics=10):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(vectorized_texts)
    return lda
