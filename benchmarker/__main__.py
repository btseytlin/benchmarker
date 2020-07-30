import re
import argparse
import sys
from collections import defaultdict

import mindsdb_native
import sklearn

from benchmarker.benchmarks.boston_house_prices import BostonMindsdbBenchmark, BostonSklearnNaiveBenchmark
from benchmarker.report import BenchmarkingReport


def run_benchmarks(benchmarks, repeat):
    results = defaultdict(dict)

    for benchmark_class in benchmarks:
        results[benchmark_class.dataset_class.name][benchmark_class.name] = []
        for i in range(repeat):
            results[benchmark_class.dataset_class.name][benchmark_class.name].append(benchmark_class().run())

    return results


def setup_args():
    parser = argparse.ArgumentParser(description='Run mindsdb benchmarks')
    parser.add_argument('--regexp', type=str, help='Run benchmarks where name matches regexp', default=None)
    parser.add_argument('--json-output-file', type=str, default=None)
    parser.add_argument('--repeat', type=int, help='How many times to run each benchmark', default=3)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    benchmarks = [
        BostonMindsdbBenchmark,
        BostonSklearnNaiveBenchmark,
    ]

    args = setup_args()

    if args.regexp:
        benchmarks = [benchmark for benchmark in benchmarks if re.match(args.regexp, benchmark.name)]

    results = run_benchmarks(benchmarks, repeat=args.repeat)

    report = BenchmarkingReport(results, args=sys.argv[1:],
                                mindsdb_version=mindsdb_native.__version__,
                                sklearn_version=sklearn.__version__)

    print(report)

    if args.json_output_file:
        with open(args.json_output_file, 'w') as f:
            f.write(report.to_json())
