window.BENCHMARK_DATA = {
  "lastUpdate": 1702975077019,
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
          "id": "f284e11d80f133602b0882dccae06a7a7a4c9c0d",
          "message": "Change BaseRunModel to throw an exception with stacktrace\n\nPreviously, if an exception occurred during ensemble evaluation, ERT\nwould throw an exception but with no stack trace. Instead, we save the\nexception instead of just the message.",
          "timestamp": "2023-12-14T14:02:35+01:00",
          "tree_id": "d37f93ae7c9fd82c60125ba54db85905febf77f1",
          "url": "https://github.com/equinor/ert/commit/f284e11d80f133602b0882dccae06a7a7a4c9c0d"
        },
        "date": 1702559127475,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.600243292228393,
            "unit": "iter/sec",
            "range": "stddev: 0.0008775778164274408",
            "extra": "mean: 151.5095664999914 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "frodeaarstad@gmail.com",
            "name": "Frode Aarstad",
            "username": "frode-aarstad"
          },
          "committer": {
            "email": "frodeaarstad@gmail.com",
            "name": "Frode Aarstad",
            "username": "frode-aarstad"
          },
          "distinct": true,
          "id": "25bd4392f15f6f7710cdd3fdf1f9f77574a53497",
          "message": "Fix bug in run analysis tool",
          "timestamp": "2023-12-15T09:38:17+01:00",
          "tree_id": "6f74c6795cb94156b16ce2c070e537517647ab40",
          "url": "https://github.com/equinor/ert/commit/25bd4392f15f6f7710cdd3fdf1f9f77574a53497"
        },
        "date": 1702629660658,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.671598215701105,
            "unit": "iter/sec",
            "range": "stddev: 0.0010295834053440922",
            "extra": "mean: 149.88912216664593 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "parulek@gmail.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "committer": {
            "email": "jparu@equinor.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "distinct": true,
          "id": "38348cd7749ec4d5a4bd2a4ed161c5c720c9e1a6",
          "message": "Add basic retry loop to account for max_submit functionality\n\nUse while retry to iterate from running to waiting states. It includes a\nsimple test to check if job has started several times. Max_submit is a function\nparameter of job.__call__ that is passed on from scheduler.\n\nAdditionally, function driver.finish will implement the basic clean up\nfunctionally. For the local driver it makes sure that all tasks have\nbeen awaited correctly.",
          "timestamp": "2023-12-15T12:48:33+01:00",
          "tree_id": "625db6daa516d7b31360308612619a8aa5f72808",
          "url": "https://github.com/equinor/ert/commit/38348cd7749ec4d5a4bd2a4ed161c5c720c9e1a6"
        },
        "date": 1702641060518,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.601358059765836,
            "unit": "iter/sec",
            "range": "stddev: 0.000626513292410273",
            "extra": "mean: 151.48398116666803 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yngve.s.kristiansen@webstep.no",
            "name": "Yngve S. Kristiansen"
          },
          "committer": {
            "email": "yngve-sk@users.noreply.github.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "distinct": true,
          "id": "e4230ffb017ba286e262f34e289558044daeaeba",
          "message": "Build wheels for ARM arch for python>=3.10",
          "timestamp": "2023-12-15T14:44:32+01:00",
          "tree_id": "1ad34cd882ad9aefadd6f02240d3a440c0147205",
          "url": "https://github.com/equinor/ert/commit/e4230ffb017ba286e262f34e289558044daeaeba"
        },
        "date": 1702648019776,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6160495577127785,
            "unit": "iter/sec",
            "range": "stddev: 0.0003148438163942655",
            "extra": "mean: 151.1475981666782 msec\nrounds: 6"
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
          "id": "e867409e33ece8937a81eb5ca8f0209ab4faee98",
          "message": "Fix off-by-one error in split_by_batchsize\n\nIf the batch size is equal to the number of parameters,\nwe want _split_by_batchsize to return a single batch and\nnot two.\n\nUpdate batch_size calculation to make sure the new\n_split_by_batchsize works when the number of parameters\nis less than the hard-coded batch_size of 1000.",
          "timestamp": "2023-12-18T09:46:25+01:00",
          "tree_id": "a42efadd18b8564f728b3dc3fabcd1945febc9ba",
          "url": "https://github.com/equinor/ert/commit/e867409e33ece8937a81eb5ca8f0209ab4faee98"
        },
        "date": 1702889370000,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.561175137785723,
            "unit": "iter/sec",
            "range": "stddev: 0.0033144453040387543",
            "extra": "mean: 152.41172183333637 msec\nrounds: 6"
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
          "id": "af717e21b46234306117e9e8693e81b3a84cf60c",
          "message": "Implement MockDriver\n\nLocalDriver differs from the HPC drivers in that it is made of three\nparts that are run in sequence. `init` the subprocess, `wait` for the\nprocess to complete and `kill` when the user wants to cancel. Between\nthe three parts the driver needs to send `JobEvent`s to the `Scheduler`.\n\nThis commit implements a `MockDriver`, where the user can optionally\nspecify a simplified version of each of `init`, `wait` or `kill`,\ndepending on what they wish to do.",
          "timestamp": "2023-12-18T10:43:09+01:00",
          "tree_id": "46cb1e2df4a761b6424f0d3fad9d0d7e6335faca",
          "url": "https://github.com/equinor/ert/commit/af717e21b46234306117e9e8693e81b3a84cf60c"
        },
        "date": 1702892740830,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.790263749088456,
            "unit": "iter/sec",
            "range": "stddev: 0.001985247527485196",
            "extra": "mean: 147.26968449999353 msec\nrounds: 6"
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
          "id": "6e051611e085ed9e64573264faab06a2af3e9ab0",
          "message": "Test unit_tests/cli with queue and scheduler",
          "timestamp": "2023-12-18T15:37:35+01:00",
          "tree_id": "207d2b5017ceebaf5cb1420ec581c27ab6120d28",
          "url": "https://github.com/equinor/ert/commit/6e051611e085ed9e64573264faab06a2af3e9ab0"
        },
        "date": 1702910406165,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.72271180103249,
            "unit": "iter/sec",
            "range": "stddev: 0.0025389817122266865",
            "extra": "mean: 148.74949716666683 msec\nrounds: 6"
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
          "id": "47e0d096e43b87d0adc2313c8cd0bc70c27b83f6",
          "message": "Raise error when loading param via response api",
          "timestamp": "2023-12-19T08:39:17+01:00",
          "tree_id": "6f021b97fc97ae5f1e5f82b2eef041d84990fe57",
          "url": "https://github.com/equinor/ert/commit/47e0d096e43b87d0adc2313c8cd0bc70c27b83f6"
        },
        "date": 1702971701627,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.800498119712296,
            "unit": "iter/sec",
            "range": "stddev: 0.002500172696027081",
            "extra": "mean: 147.0480518333422 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "eide.oyvind87@gmail.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "committer": {
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "d3b6ae4540c8a6fa12acfbe5cd584428510373a7",
          "message": "Remove global enkf_main",
          "timestamp": "2023-12-19T09:35:07+01:00",
          "tree_id": "58e1ec285850ec6b528340f396858a285bcf3d72",
          "url": "https://github.com/equinor/ert/commit/d3b6ae4540c8a6fa12acfbe5cd584428510373a7"
        },
        "date": 1702975076579,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.284767443480779,
            "unit": "iter/sec",
            "range": "stddev: 0.029019840798108605",
            "extra": "mean: 159.11487719999968 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}