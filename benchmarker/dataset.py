from sklearn.model_selection import KFold
from abc import ABC


class Dataset(ABC):
    name = None

    def __init__(self, full_df=None,
                 train_df=None,
                 test_df=None):
        self.full_df = full_df
        self.train_df = train_df
        self.test_df = test_df

    def setup(self):
        pass
