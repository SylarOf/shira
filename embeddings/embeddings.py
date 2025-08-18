import numpy as np
def embed_document(self,text:str)->list[float]:
    ""    
    
    
def embed_query(self,text:str) ->list[float]:
    ""
    
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))