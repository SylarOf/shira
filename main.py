from rag import Rag
from rag import MessageWarp

def start(question):
    initial_state : MessageWarp = (question,None)  # type: ignore
    MessageWarp = Rag(initial_state)
    
    while(1):
        
