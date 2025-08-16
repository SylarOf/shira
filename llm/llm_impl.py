from .import LLMInfer

class LLMInferImpl(LLMInfer):
    def __init__(self,paras):
        "todo!"
    
    def extract_key_message_from_questrion(self,question)->str:
        ""
    def extract_kg_from_answer(self,answer)->KG:
        "" 

    def extract_kg_from_file(self,file)->KG:
        ""
    
    def extract_entities_from_question(self,question):
        ""
    
    def infer_from_questrion_and_kg(self,question,kg)->str:
        ""
    
    def get_abstract_from_file(self,file):
        ""


def new_llm_infer(paras)->LLMInfer:
    return LLMInfer(paras)