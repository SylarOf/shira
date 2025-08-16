from abc import ABC,abstractmethod
from typing import TYPE_CHECKING

class DataSet(ABC):
    @abstractmethod
    def __len__(self):
        ""
    
    @abstractmethod
    def __getitem__(self,index):
        ""
    
    @abstractmethod
    def load_data(self):
        ""

if TYPE_CHECKING:
    def new_dataset(data)->DataSet:...