from .import DataSet

class DataSetImpl(DataSet):
    def __init__(self,data):
        self.data = data
        self.load_data()
    
    def load_data(self):
        """logic to load data

        func(self.data)"""

    def __len__(self):
        "method to get len of data" 

    def __getitem__(self, index):
        "impl DataSetImpl[index]"

def new_dataset(data)->DataSet:
    return DataSetImpl(data)