window.BENCHMARK_DATA = {
  "lastUpdate": 1702473210316,
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
      },
      {
        "commit": {
          "author": {
            "email": "feda.curic@gmail.com",
            "name": "Feda Curic",
            "username": "dafeda"
          },
          "committer": {
            "email": "feda.curic@gmail.com",
            "name": "Feda Curic",
            "username": "dafeda"
          },
          "distinct": true,
          "id": "374c4b70cd77f7571377976e5a42c84bd108a9a5",
          "message": "Document adaptive localization",
          "timestamp": "2023-12-12T13:29:24+01:00",
          "tree_id": "f5c0efa981595a52c4401ffa5269e3a8f07f21af",
          "url": "https://github.com/equinor/ert/commit/374c4b70cd77f7571377976e5a42c84bd108a9a5"
        },
        "date": 1702384327684,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.217316824460933,
            "unit": "iter/sec",
            "range": "stddev: 0.0015043269760327653",
            "extra": "mean: 160.841087599988 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "lundeberent@gmail.com",
            "name": "Berent Å. S. Lunde",
            "username": "Blunde1"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fd4b90279ad6ee09c3c196b7f9c6ac096510bfe6",
          "message": "Use new API from iterative_ensemble_smoother (#6634)\n\n* Use v.0.2.0 of iterative_ensemble_smoother\r\n\r\n* Update analysis module to use new API from iterative_ensemble_smoother.\r\nThis affects ES, ES-MDA, IES, adaptive localization, and row-scaling.\r\n\r\nCo-authored-by: Tommy Odland <tommy.odland>",
          "timestamp": "2023-12-13T02:39:30-08:00",
          "tree_id": "afcad5a4a98add12dcbd213083bf7ffd5c445eda",
          "url": "https://github.com/equinor/ert/commit/fd4b90279ad6ee09c3c196b7f9c6ac096510bfe6"
        },
        "date": 1702464143378,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.287644257160445,
            "unit": "iter/sec",
            "range": "stddev: 0.024250420786383328",
            "extra": "mean: 159.04207666666062 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "havb@equinor.com",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "committer": {
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "c3d9e34d463d263875b3d21fb1ed0ee5dbf6e9fa",
          "message": "Let the Scheduler ensemble be stoppable from evaluator.py\n\nFor some reason, kill_all_jobs() was not able to kill tasks,\nseemingly the await self.returncode call was blocking. Solved\nby \"busy waiting\" with asyncio.sleep to let the async code accept\nthe cancellation.",
          "timestamp": "2023-12-13T14:10:57+01:00",
          "tree_id": "039b0c5d842c2508095f088c17ff15f7487a08f3",
          "url": "https://github.com/equinor/ert/commit/c3d9e34d463d263875b3d21fb1ed0ee5dbf6e9fa"
        },
        "date": 1702473209899,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.49684861587087,
            "unit": "iter/sec",
            "range": "stddev: 0.00208765481828311",
            "extra": "mean: 153.9207790000129 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}