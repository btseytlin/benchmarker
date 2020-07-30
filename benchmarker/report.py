import datetime
import json
from collections import defaultdict
import numpy as np


class BenchmarkingReport:
    def __init__(self, results, args,
                 mindsdb_version,
                 sklearn_version,
                 dt=None):
        self.results = results
        self.datetime = dt or datetime.datetime.utcnow()
        self.args = args
        self.mindsdb_version = mindsdb_version
        self.sklearn_version = sklearn_version

    def __str__(self):
        lines = list()
        lines.append(self.datetime.isoformat())
        lines.append(f'mindsdb_native {self.mindsdb_version}')
        lines.append(f'sklearn {self.sklearn_version}')
        lines.append(f'Ran with args: {self.args}')
        for dataset, dataset_benchmarks in self.results.items():
            lines.append(dataset)
            lines.append('---')
            for benchmark in dataset_benchmarks:
                results = dataset_benchmarks[benchmark]
                lines.append(f'{benchmark}')

                scores = defaultdict(list)
                for result in results:
                    for score, score_value in result.scores.items():
                        scores[score].append(score_value)

                for score in scores:
                    lines.append(f'\t{score}_mean: {round(np.mean(scores[score]), 5)}')
                    lines.append(f'\t{score}_std: {round(np.std(scores[score]), 5)}')
            lines.append('\n')
        return '\n'.join(lines)

    def to_dict(self):
        out_dict = {}
        out_dict['args'] = self.args
        out_dict['datetime'] = self.datetime.isoformat()
        out_dict['mindsdb_version'] = self.mindsdb_version
        out_dict['sklearn_version'] = self.sklearn_version

        out_dict['benchmarks'] = defaultdict(dict)
        for dataset, dataset_benchmarks in self.results.items():
            for benchmark in dataset_benchmarks:
                results = dataset_benchmarks[benchmark]

                scores = defaultdict(list)
                for result in results:
                    for score, score_value in result.scores.items():
                        scores[score].append(score_value)

                out_dict['benchmarks'][dataset][benchmark] = {}
                for score in scores:
                    score_mean = np.mean(scores[score])
                    score_std = np.std(scores[score])

                    out_dict['benchmarks'][dataset][benchmark][score] = {
                        'values': scores[score],
                        f'mean': score_mean,
                        f'std': score_std,
                    }
        return out_dict

    def to_json(self):
        out_dict = self.to_dict()
        json_string = json.dumps(out_dict, indent=4)
        return json_string
