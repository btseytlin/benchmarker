from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from mindsdb_native import Predictor
from benchmarker.benchmark import Benchmark, MindsdbBenchmark, BenchmarkResult
from benchmarker.datasets.boston_house_prices import BostonHousePricesDataset


class BostonBenchmark(Benchmark):
    target_column = 'MEDV'
    score_name = 'r2_score'
    dataset_class = BostonHousePricesDataset

    def get_train_test_split(self):
        train_df, test_df = train_test_split(self.dataset.full_df)
        return train_df, test_df

    def evaluate_score(self, true_values, predictions):
        return r2_score(true_values, predictions)


class BostonMindsdbBenchmark(BostonBenchmark, MindsdbBenchmark):
    name = 'boston_mindsdb'

    def train(self, train_df):
        mdb = Predictor(name=self.name)
        mdb.learn(to_predict=self.target_column,
                  from_data=train_df,
                  backend='lightwood')
        return mdb

    def predict(self, mdb, test_df):
        predictions = mdb.predict(when_data=test_df)

        predicted_values = []
        for i in range(len(predictions)):
            predicted_values.append(predictions[i][self.target_column])

        return predicted_values


class BostonSklearnNaiveBenchmark(BostonBenchmark):
    name = 'boston_sklearn_naive'

    def train(self, train_df):
        model = RandomForestRegressor(n_jobs=-1)
        train_X = train_df.drop(columns=self.target_column)
        train_Y = train_df[self.target_column]
        model.fit(train_X, train_Y)
        return model

    def predict(self, model, test_df):
        predictions = model.predict(test_df)

        return predictions
