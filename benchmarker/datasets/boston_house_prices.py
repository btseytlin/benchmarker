import pandas as pd
from sklearn.datasets import load_boston
from benchmarker.dataset import Dataset


class BostonHousePricesDataset(Dataset):
    """https://scikit-learn.org/stable/datasets/index.html#boston-dataset"""
    name = 'boston_house_prices'

    def setup(self):
        boston_bunch = load_boston()

        feature_names = boston_bunch.feature_names
        df = pd.DataFrame(boston_bunch.data, columns=feature_names)
        df['MEDV'] = boston_bunch.target

        self.full_df = df
