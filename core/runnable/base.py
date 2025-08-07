from typing import Any,Dict,List
from __future__ import annotations

class Runnable:
    def invoke(self,input:Any)->Any:
        
        raise NotImplementedError
    
    def __or__(self, other:Runnable)->Chain:
        return Chain(steps=[self,other]) 
    
    def __repr__(self)->str:
        return f"<{self.__class__.__name__}>"


class Chain(Runnable):
    def __init__(self, steps:List[Runnable]):
        self.steps = steps

    def invoke(self, input:Any)->Any:
        current = input
        print(f"start executing, input: {current}\n")
        for i, step in enumerate(self.steps):
            print(f"step: {step}\n")
            current = step.invoke(current)
            print(f"step: {step} output is {current}\n")
        
        return current

    def __or__(self, other:Runnable)->Chain:
        return Chain(self.steps+[other])
    
    def __repr__(self):
        return " | ".join([repr(step) for step in self.steps])
    