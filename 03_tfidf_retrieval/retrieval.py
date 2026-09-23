import numpy as np

from text_utils import tokenize
from tfidf import cosine_scores, vectorize_query


def rank_documents(query: str, documents: list[str], tokenized_documents: list[list[str]], vocabulary: dict[str, int], idf, matrix, top_k: int = 5):
    query_vector = vectorize_query(tokenize(query), vocabulary, idf)
    scores = cosine_scores(query_vector, matrix)
    order = np.argsort(-scores)[:top_k]
    return [(int(index), float(scores[index]), documents[int(index)]) for index in order]
