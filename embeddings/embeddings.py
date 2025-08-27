from sentence_transformers import SentenceTransformer, util
model_name = "Qwen/Qwen3-Embedding-0.6B"
def embed_document(text:str)->list[float]:
    emb_model = SentenceTransformer(model_name)
    embedding = emb_model.encode(text)
    return embedding


def embed_query(text:str) ->list[float]:
    emb_model = SentenceTransformer(model_name)
    embedding = emb_model.encode(text)
    # print(len(embedding))
    # print(emb_model.get_sentence_embedding_dimension())
    return embedding

def cosine_similarity(a, b):
    return util.cos_sim(a, b)


if __name__ == "__main__":
    sentence1 = "this is an apple"
    sentence2 = "this is an big apple"
    sentence3 = "The universe is very large."

    emb1 = embed_query(sentence1)
    emb2 = embed_query(sentence2)
    emb3 = embed_query(sentence3)

    print(emb1)
    print(emb2)
    print(emb3)

    score1 = cosine_similarity(emb1, emb2)
    score2 = cosine_similarity(emb1, emb3)
    score3 = cosine_similarity(emb2, emb3)
    print(score1)
    print(score2)
    print(score3)
