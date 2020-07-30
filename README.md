# Running
Defaults:
`python benchmarker`

Benchmarks by regexp:

`python benchmarker --regexp .*sklearn.*`

Report to json file:

`python benchmarker --json-output-file report.json`

Run each benchmark multiple times and aggregate results (by default 3):

`python benchmarker --repeat 3`

# Example console report
```
2020-07-30T16:57:34.113747
mindsdb_native 2.1.0
sklearn 0.23.1
Ran with args: []

boston_house_prices
---
boston_mindsdb
        r2_score_mean: 0.70345
        r2_score_std: 0.02091
        time_train_mean: 29.68492
        time_train_std: 0.67738
        time_predict_mean: 0.07621
        time_predict_std: 0.00232
boston_sklearn_naive
        r2_score_mean: 0.88913
        r2_score_std: 0.02887
        time_train_mean: 0.11432
        time_train_std: 0.00096
        time_predict_mean: 0.01126
        time_predict_std: 0.00017
```

# Core principles

## 1. Code enforces fair benchmarks

The class structure of benchmarks forces each model to receive data splits in the same way. 
This reduces the possibility of a developer introducing an unfair benchmark.

## 2. Code is as easy to extend as possible

It follows the [SOLID](https://en.wikipedia.org/wiki/SOLID ), 
[KISS](https://en.wikipedia.org/wiki/KISS_principle) 
and [YAGNI](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it) principles.

It's fairly easy to add any dataset. 
Extending the suite (for example to support multiple scoring functions per dataset) requires very few modifications.
The modular structure makes it trivial to distribute or parallelize the benchmarks.

## 3. Fail fast

There is no exception-swallowing. If something is broken, you will find out very soon and have a chance to fix it.

## 4. Each benchmark runs multiple times

It's important to monitor the mean score across multiple runs because models can achieve different results due to randomness or different data splits.
Monitoring the standard deviation is also useful because if it increased after some code change that
 means the model became more sensitive to the data splits and less robust overall.
 

