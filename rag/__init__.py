from runnable import Runnable
from kg import KG, Triple
from llm import LLMInfer
from typing import Tuple
from embeddings import Embeddings 


MessageWarp = Tuple[str, KG]

class Rag(Runnable[MessageWarp,MessageWarp]):
    def invoke(self, MessageWarp) ->MessageWarp:
        question, kg = MessageWarp 

        question = LLMInfer.extract_key_message_from_questrion(question)

        files = Embeddings.add_files_form_query(question)

        kg = LLMInfer.extract_kg_from_file(files)

        entities = LLMInfer.extract_entities_from_question(question)

        sub_graph = self.get_subgraph(entities=entities)

        kg.add_triples(sub_graph)

        answer = LLMInfer.infer_from_questrion_and_kg(question,kg)

        answer_kg = LLMInfer.extract_kg_from_answer(answer)

        kg = kg.append(answer_kg)

        return (answer,kg)

        
    
    def get_subgraph(self,entities) ->list[Triple]:
        ""

