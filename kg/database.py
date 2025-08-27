import numpy as np
from typing import Dict,List,Optional
from sklearn.metrics.pairwise import cosine_similarity
from rocksdict import Rdict
from embeddings.embeddings import embed_query

import faiss
import numpy as np
import pickle
import os

class VectorDB:
    def __init__(self, dim=512, embedding_func=None, db_path="vectordb.pkl"):
        self.dim = dim
        self.embedding_func = embedding_func
        self.db_path = db_path
        
        # FAISS index
        self.index = faiss.IndexFlatL2(dim)  # L2 距离
        self.keys = []   # 保存对应 key/query
        self.vectors = []  # 保存向量，用于持久化
        
        # 加载已有数据库
        if os.path.exists(db_path):
            self.load()

    def insert(self, query):
        vector = np.array(self.embedding_func(query), dtype='float32').reshape(1, -1)
        self.index.add(vector)
        self.keys.append(query)
        self.vectors.append(vector)
        self.save()

    def get(self, query):
        # 精确查找 query
        if query in self.keys:
            idx = self.keys.index(query)
            return self.vectors[idx].flatten()
        return None

    def search(self, query, top_k=5):
        vector = np.array(self.embedding_func(query), dtype='float32').reshape(1, -1)
        D, I = self.index.search(vector, top_k)  # D: distances, I: indices
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.keys):
                results.append((self.keys[idx], dist))
        return results

    def save(self):
        # 保存 index + keys
        faiss.write_index(self.index, self.db_path + ".index")
        with open(self.db_path + ".meta", "wb") as f:
            pickle.dump(self.keys, f)

    def load(self):
        self.index = faiss.read_index(self.db_path + ".index")
        with open(self.db_path + ".meta", "rb") as f:
            self.keys = pickle.load(f)
        # 重建 vectors 列表
        self.vectors = [self.index.reconstruct(i).reshape(1, -1) for i in range(len(self.keys))]
