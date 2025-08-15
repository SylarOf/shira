from abc import ABC, abstractmethod

class Embeddings(ABC):
    @abstractmethod
    def embed_documents(self,text:list[str])->list[list[float]]:
        ""    
    
    @abstractmethod
    def embed_query(self,text:str) ->list[float]:
        ""
    
    @abstractmethod
    def add_files_form_query(self,text:str,files:list[str]) ->list[str]:
        ""