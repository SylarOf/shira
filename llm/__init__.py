from transformers import AutoTokenizer,AutoModelForCausalLM
from typing import TYPE_CHECKING
from abc import ABC,abstractmethod
from kg import KG

class LLMInfer(ABC):
    @abstractmethod
    def extract_key_message_from_questrion(self,question)->str:
        ""
    @abstractmethod
    def extract_kg_from_answer(self,answer)->KG:
        "" 

    @abstractmethod
    def extract_kg_from_file(self,file)->KG:
        ""
    
    @abstractmethod
    def extract_entities_from_question(self,question):
        ""
    
    @abstractmethod
    def infer_from_questrion_and_kg(self,question,kg)->str:
        ""
    
    @abstractmethod
    def get_abstract_from_file(self,file):
        ""

    @abstractmethod
    def get_topk_files_from_questrion(self,questrion,files):
        ""


if TYPE_CHECKING:
    "i don't know paras are"
    def new_llm_infer(paras)->LLMInfer:...