from abc import ABC,abstractmethod
from typing import Hashable,Tuple,Iterable

Entity = Hashable
Predicate = Hashable
Triple = Tuple[Entity,Predicate,Entity]

class KG(ABC):
    @abstractmethod
    def add_triple(self,triple):
        ""
   
    @abstractmethod
    def add_triples(self,triples):
        ""

    @abstractmethod
    def append(self, kg):
        ""
    
    @abstractmethod
    def flush(self):
        ""