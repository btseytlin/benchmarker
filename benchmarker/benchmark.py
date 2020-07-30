import time
import shutil
import tempfile
import mindsdb_native
from abc import ABC


class BenchmarkResult(ABC):
    def __init__(self, benchmark, scores):
        self.benchmark = benchmark
        self.scores = scores


class Benchmark(ABC):
    name = None
    dataset_class = None
    target_column = None
    score_name = None

    def __init__(self, dataset=None):
        self.dataset = dataset

    def setup(self):
        self.dataset = self.dataset_class()
        self.dataset.setup()

    def get_train_test_split(self):
        pass

    def evaluate_score(self, true_values, predictions):
        pass

    def cleanup(self):
        pass

    def train(self, train_df):
        pass

    def predict(self, model, test_df):
        pass

    def run(self):
        self.setup()

        train_df, test_df = self.get_train_test_split()

        training_time = time.time()
        model = self.train(train_df)
        training_time = time.time() - training_time

        test_df_no_target = test_df.drop(columns=[self.target_column])

        predictions_time = time.time()
        predictions = self.predict(model, test_df_no_target)
        predictions_time = time.time() - predictions_time

        score = self.evaluate_score(test_df[self.target_column], predictions)

        result = BenchmarkResult(
            benchmark=self,
            scores={self.score_name: score,
                    'time_train': training_time,
                    'time_predict': predictions_time}
        )
        return result


class MindsdbBenchmark(Benchmark):
    def setup(self):
        super().setup()
        mindsdb_native.config.CONFIG.CHECK_FOR_UPDATES = False
        # Remove cached mindsdb stuff for a clean run
        self.tmp_dir = tempfile.mkdtemp()
        mindsdb_native.config.CONFIG.MINDSDB_STORAGE_PATH = self.tmp_dir

    def cleanup(self):
        super().cleanup()
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception as e:
            print(e)
            pass