window.BENCHMARK_DATA = {
  "lastUpdate": 1702557370439,
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
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "b2b2e4876731da03aaf5491bf9cbeae3bf689b26",
          "message": "Add \"done_callback\" to every task for debugging\n\nCurrently, tasks that throw exceptions silently stop. This commit makes\nit so that all tasks are created with a done_callback that checks\nwhether they stopped due to an exception.",
          "timestamp": "2023-12-13T14:30:41+01:00",
          "tree_id": "d067a7926baea28dfc141c61331a11642c503636",
          "url": "https://github.com/equinor/ert/commit/b2b2e4876731da03aaf5491bf9cbeae3bf689b26"
        },
        "date": 1702474403771,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.639348653516157,
            "unit": "iter/sec",
            "range": "stddev: 0.0013389145468599459",
            "extra": "mean: 150.61718433334667 msec\nrounds: 6"
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
          "id": "225a1e8ea6ba5d990ccd161674388467423a7607",
          "message": "Let publisher_task exit gracefully when nothing to do",
          "timestamp": "2023-12-13T15:21:22+01:00",
          "tree_id": "338f7bc3f1d49d1a197318e9ba35c548b3633893",
          "url": "https://github.com/equinor/ert/commit/225a1e8ea6ba5d990ccd161674388467423a7607"
        },
        "date": 1702477427173,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.224977477486739,
            "unit": "iter/sec",
            "range": "stddev: 0.02677738237736511",
            "extra": "mean: 160.6431515000016 msec\nrounds: 6"
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
          "id": "f3063c40a32d824013017a8c34d19dde71e401c2",
          "message": "Raise if params not registered are saved",
          "timestamp": "2023-12-14T12:20:32+01:00",
          "tree_id": "6f495fff2fdb0b992ad1b9f74e40809a54d53f41",
          "url": "https://github.com/equinor/ert/commit/f3063c40a32d824013017a8c34d19dde71e401c2"
        },
        "date": 1702553010170,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.33603552822918,
            "unit": "iter/sec",
            "range": "stddev: 0.019759675605222914",
            "extra": "mean: 157.82739783333946 msec\nrounds: 6"
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
          "id": "174f548964fdf6a481587c40afb5c257999e85bb",
          "message": "Only perturb observations once\n\nNo need to perturb per parameter batch",
          "timestamp": "2023-12-14T12:33:25+01:00",
          "tree_id": "cc71b61115538f42ba43bf722b73f047083d9865",
          "url": "https://github.com/equinor/ert/commit/174f548964fdf6a481587c40afb5c257999e85bb"
        },
        "date": 1702553781807,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.594515638755547,
            "unit": "iter/sec",
            "range": "stddev: 0.0006281092751751284",
            "extra": "mean: 151.6411598333415 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "JONAK@equinor.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "committer": {
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "07818cfda2559facd01b183528efb4b2696c4504",
          "message": "Add fixture for running tests with scheduler and job queue",
          "timestamp": "2023-12-14T12:54:38+01:00",
          "tree_id": "95e3eb8975a991332f5d31790dc6591b73eea3d7",
          "url": "https://github.com/equinor/ert/commit/07818cfda2559facd01b183528efb4b2696c4504"
        },
        "date": 1702555032199,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.563279079685446,
            "unit": "iter/sec",
            "range": "stddev: 0.0009047371190215223",
            "extra": "mean: 152.36286433334575 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "65c62c60284a9e9e18c5790e6782f3d7aaac3117",
          "message": "Add `background_tasks` for canceling tasks\n\nThis contextmanager makes it simpler to ensure that infinitely-running\ntasks get canceled when appropriate, even when an exception occurs.",
          "timestamp": "2023-12-14T13:33:36+01:00",
          "tree_id": "e9e461dc7e37f261951569b359ed35d43b32f0f9",
          "url": "https://github.com/equinor/ert/commit/65c62c60284a9e9e18c5790e6782f3d7aaac3117"
        },
        "date": 1702557370024,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.607344940034548,
            "unit": "iter/sec",
            "range": "stddev: 0.001156330428637273",
            "extra": "mean: 151.34672233333882 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}