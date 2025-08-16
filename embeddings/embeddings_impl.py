from .import Embeddings

class EmbeddingsImpl(Embeddings):
    def __init__(self):
        ""
        
    def embed_documents(self,text:list[str])->list[list[float]]:
        ""    
    
    
    def embed_query(self,text:str) ->list[float]:
        ""
    
    
    def add_topk_files_form_query(self,text:str,files:list[str]) ->list[str]:
        ""

def new_embeddings()->Embeddings:
    "i don't konw"
    return EmbeddingsImpl