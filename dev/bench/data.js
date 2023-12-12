window.BENCHMARK_DATA = {
  "lastUpdate": 1702371332810,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "feda.curic@gmail.com",
            "name": "Feda Curic",
            "username": "dafeda"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "050d02afefd02427c0558ead4916c8a527a989f6",
          "message": "Add GitHub workflow for benchmarking",
          "timestamp": "2023-12-12T09:52:58+01:00",
          "tree_id": "9bf332bde5e48fdd3036fec8c8285e856458d6a7",
          "url": "https://github.com/equinor/ert/commit/050d02afefd02427c0558ead4916c8a527a989f6"
        },
        "date": 1702371332232,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.182927578442731,
            "unit": "iter/sec",
            "range": "stddev: 0.0012028127090517138",
            "extra": "mean: 161.7356806000089 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}