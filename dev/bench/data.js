window.BENCHMARK_DATA = {
  "lastUpdate": 1733478014527,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "08e98c5794620522227654da5a821fba9ba475e8",
          "message": "Move performance tests out of unit_tests",
          "timestamp": "2024-12-02T11:04:02+01:00",
          "tree_id": "87e51ac6fcf674d05465b0116fad86c4d9ce4b3b",
          "url": "https://github.com/equinor/ert/commit/08e98c5794620522227654da5a821fba9ba475e8"
        },
        "date": 1733133957145,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19138848497139124,
            "unit": "iter/sec",
            "range": "stddev: 0.046897176649856",
            "extra": "mean: 5.224974742600006 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dan.sava42@gmail.com",
            "name": "DanSava",
            "username": "DanSava"
          },
          "committer": {
            "email": "dan.sava42@gmail.com",
            "name": "Dan Sava",
            "username": "DanSava"
          },
          "distinct": true,
          "id": "0d1430ed21320779649478c0617ec0401260d722",
          "message": "Add also the simulation jobs to the list of possible simulator instances.\n\nEverest is adding SIMULATION_JOB key and not FORWARD_MODEL key for the steps in the forward model",
          "timestamp": "2024-12-02T19:16:01+09:00",
          "tree_id": "be0fcbea412d0eaf4ee25c7aaf752e3bad712685",
          "url": "https://github.com/equinor/ert/commit/0d1430ed21320779649478c0617ec0401260d722"
        },
        "date": 1733134672326,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19036200242006743,
            "unit": "iter/sec",
            "range": "stddev: 0.036633390621088266",
            "extra": "mean: 5.253149196200002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dan.sava42@gmail.com",
            "name": "Dan Sava",
            "username": "DanSava"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b5a1805e6517c52fc9007b0e4cd7ed862c452fe0",
          "message": "Use ensemble evaluator in everest in place of BatchSimulator\n\n* Use ensemble evaluator in everest in place of BatchSimulator\r\n\r\n* Fix test_simulator_cache Co-authored-by: Peter Verveer <pieter.verveer@tno.nl>",
          "timestamp": "2024-12-02T12:17:55+02:00",
          "tree_id": "4b0da21e8f1407f01ded1dbb2640cc45547c5c36",
          "url": "https://github.com/equinor/ert/commit/b5a1805e6517c52fc9007b0e4cd7ed862c452fe0"
        },
        "date": 1733134789517,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1973472987219686,
            "unit": "iter/sec",
            "range": "stddev: 0.04097912290169314",
            "extra": "mean: 5.067208958399999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "c9fc8d30ef52b8b98c8c3ff9d37f805cbbdbcc6d",
          "message": "Fix handling '/' in _get_num_cpu",
          "timestamp": "2024-12-02T12:00:56+01:00",
          "tree_id": "aaec44910500225a36278417e2e8f5ecd7c36ad5",
          "url": "https://github.com/equinor/ert/commit/c9fc8d30ef52b8b98c8c3ff9d37f805cbbdbcc6d"
        },
        "date": 1733137369426,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19181437147688954,
            "unit": "iter/sec",
            "range": "stddev: 0.02435365875604968",
            "extra": "mean: 5.213373702399997 sec\nrounds: 5"
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
            "email": "107626001+jonathan-eq@users.noreply.github.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "distinct": true,
          "id": "ef9b3b8e3b8c6dd0981853032762742dfaa915bc",
          "message": "Fix _storage_main `terminate_on_parent_death` not working on mac\n\nPrior to this commit, the `terminate_on_parent_death` function was only usable on linux, due to it using the prctl command.\nThis commit creates a new thread which polls the parent process, and signals terminate when it can no longer find the parent.",
          "timestamp": "2024-12-02T13:04:34+01:00",
          "tree_id": "ec5397136b5122e8f34c11d9e0df672a3d93affe",
          "url": "https://github.com/equinor/ert/commit/ef9b3b8e3b8c6dd0981853032762742dfaa915bc"
        },
        "date": 1733141191570,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18882702036145596,
            "unit": "iter/sec",
            "range": "stddev: 0.03441055395684877",
            "extra": "mean: 5.295852246599997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "2ef40b4278dd51c4053731283dc8971dda821230",
          "message": "Make it possible to have multiline statements in config",
          "timestamp": "2024-12-02T13:23:38+01:00",
          "tree_id": "3de8fba5f73c16034f691bb7f73045152a5f2937",
          "url": "https://github.com/equinor/ert/commit/2ef40b4278dd51c4053731283dc8971dda821230"
        },
        "date": 1733142333408,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19146370145251476,
            "unit": "iter/sec",
            "range": "stddev: 0.029801264624543234",
            "extra": "mean: 5.222922112200007 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3a84a98e4447dbf7101d8661fbfcc6ee525181f6",
          "message": "Update ropt to 0.10.0 (#9405)\n\n* Update ropt to 0.10.0\r\n\r\n* Run ruff format",
          "timestamp": "2024-12-02T13:25:53+01:00",
          "tree_id": "a0364d4dc7359bc7379ab7a1db23f87199cfa99f",
          "url": "https://github.com/equinor/ert/commit/3a84a98e4447dbf7101d8661fbfcc6ee525181f6"
        },
        "date": 1733142465450,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19563369033055503,
            "unit": "iter/sec",
            "range": "stddev: 0.03777665863641594",
            "extra": "mean: 5.111594011799997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ynk@equinor.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "committer": {
            "email": "yngve-sk@users.noreply.github.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "distinct": true,
          "id": "22687ea7c012ab36176ad48cff53884726f55273",
          "message": "Remove test_api.py\n\nBehavior of plot API is still locked in by snapshots in `test_api_snapshots.py`",
          "timestamp": "2024-12-02T15:46:46+01:00",
          "tree_id": "d4179bc83fb0e947b7c3a72289fd2a8071229cdb",
          "url": "https://github.com/equinor/ert/commit/22687ea7c012ab36176ad48cff53884726f55273"
        },
        "date": 1733150917551,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19379841736524772,
            "unit": "iter/sec",
            "range": "stddev: 0.007068278583113922",
            "extra": "mean: 5.160000858600003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ynk@equinor.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "committer": {
            "email": "yngve-sk@users.noreply.github.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "distinct": true,
          "id": "c686535222988ea6f4d6ad1bc3ad8c1e6caaf254",
          "message": "Add fixture for caching everest test-data example",
          "timestamp": "2024-12-03T08:49:40+01:00",
          "tree_id": "d03846a637490986ce2db911aa7537a7cb53b6f2",
          "url": "https://github.com/equinor/ert/commit/c686535222988ea6f4d6ad1bc3ad8c1e6caaf254"
        },
        "date": 1733212296587,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.195100683014254,
            "unit": "iter/sec",
            "range": "stddev: 0.0332584550921401",
            "extra": "mean: 5.125558683599996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "ec8b5cc05487895058c45c0d9b9db863901655f0",
          "message": "Add coverage gathering of everest",
          "timestamp": "2024-12-03T09:58:52+01:00",
          "tree_id": "1cbec251b07c0a9d82460a48da04c2fbfa95ae5e",
          "url": "https://github.com/equinor/ert/commit/ec8b5cc05487895058c45c0d9b9db863901655f0"
        },
        "date": 1733216446585,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19484407227573639,
            "unit": "iter/sec",
            "range": "stddev: 0.025283097746221977",
            "extra": "mean: 5.132309073200008 sec\nrounds: 5"
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
          "id": "eea080c93eec1b812a917f5c2016fc81f8d3f352",
          "message": "Ignore erroneous auto-scale configuration, but warn",
          "timestamp": "2024-12-03T11:43:07+01:00",
          "tree_id": "ac4f4a4ffdb746233a6165e9ae1af4b2baa72ad6",
          "url": "https://github.com/equinor/ert/commit/eea080c93eec1b812a917f5c2016fc81f8d3f352"
        },
        "date": 1733222698752,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.20023494236975783,
            "unit": "iter/sec",
            "range": "stddev: 0.052400916188259684",
            "extra": "mean: 4.994133332399997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "levje@equinor.com",
            "name": "larsevj",
            "username": "larsevj"
          },
          "committer": {
            "email": "60844986+larsevj@users.noreply.github.com",
            "name": "Lars Evje",
            "username": "larsevj"
          },
          "distinct": true,
          "id": "691b3e84de16d4dc0879931ee708f12dca4836fa",
          "message": "Upgrade setup-uv v3 -> v4",
          "timestamp": "2024-12-03T12:36:57+01:00",
          "tree_id": "b660a2e572c6e3ab66634a62c67fea4bb7f86283",
          "url": "https://github.com/equinor/ert/commit/691b3e84de16d4dc0879931ee708f12dca4836fa"
        },
        "date": 1733225926766,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1971483820287773,
            "unit": "iter/sec",
            "range": "stddev: 0.02822813196409",
            "extra": "mean: 5.072321617399996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "0ee3d4906ebf32a15be67979dea908cd3a95d4f5",
          "message": "Make local_driver use start_new_session when spawning children\n\nThis makes it more in line with lsf driver.\npreexec_fn can also deadlock when using fork method for multiprocessing\nmodule.",
          "timestamp": "2024-12-03T13:01:27+01:00",
          "tree_id": "258fc63207c210f5c9dcff0c590298ce01dc5a12",
          "url": "https://github.com/equinor/ert/commit/0ee3d4906ebf32a15be67979dea908cd3a95d4f5"
        },
        "date": 1733227396920,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19343094337051872,
            "unit": "iter/sec",
            "range": "stddev: 0.014114403390293724",
            "extra": "mean: 5.169803665199993 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "5dbaae50d8f43eb9a7af2ad139d99913439c64df",
          "message": "Update python version in benchmark workflow",
          "timestamp": "2024-12-03T13:31:46+01:00",
          "tree_id": "03285172575be2ef1295110daad16b098a3ba46c",
          "url": "https://github.com/equinor/ert/commit/5dbaae50d8f43eb9a7af2ad139d99913439c64df"
        },
        "date": 1733229220532,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21755028259359038,
            "unit": "iter/sec",
            "range": "stddev: 0.026973425817297007",
            "extra": "mean: 4.596638478599994 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "03ee39ab21d16857039cd11342728d2a16f4207b",
          "message": "Fix import that does not work in python 3.10\n\nReplace UTC with timezone.utc",
          "timestamp": "2024-12-03T14:21:13+01:00",
          "tree_id": "c6a55df98a348dcc5cc7f20ae7cee3436d9c53bc",
          "url": "https://github.com/equinor/ert/commit/03ee39ab21d16857039cd11342728d2a16f4207b"
        },
        "date": 1733232184193,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21143399951008976,
            "unit": "iter/sec",
            "range": "stddev: 0.039980224908260774",
            "extra": "mean: 4.729608304800001 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "309a5353a9bc25b7adc98a83e6fd9c5f6747eb40",
          "message": "Revert \"Use uvloop for asyncio\"\n\nThis reverts commit d11ba38bd8f93808ae60d320e03ab10725294dec.\nThis was done due to issues with uvloop when attempting to use uvloop\nfor our tests",
          "timestamp": "2024-12-03T14:22:23+01:00",
          "tree_id": "1559784effd3f73924a6b411e2abf7b36a4c6ec9",
          "url": "https://github.com/equinor/ert/commit/309a5353a9bc25b7adc98a83e6fd9c5f6747eb40"
        },
        "date": 1733232259426,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2115476281274497,
            "unit": "iter/sec",
            "range": "stddev: 0.02510521401801525",
            "extra": "mean: 4.7270678894000016 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "c4b5b67b99ed627ae149da0365e4e6593eec9c13",
          "message": "Use forkserver instead of fork for multiprocessing\n\nPolars gives warnings if using fork.\npreexec_fn in create_subprocess_exec is unsafe using fork.\npython 3.14 will set forkserver as default on linux/bsd.",
          "timestamp": "2024-12-03T15:24:43+01:00",
          "tree_id": "c413e03f43c4c7d84b340fb1350031734613a2ca",
          "url": "https://github.com/equinor/ert/commit/c4b5b67b99ed627ae149da0365e4e6593eec9c13"
        },
        "date": 1733235996444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2166364306550501,
            "unit": "iter/sec",
            "range": "stddev: 0.025677284970481586",
            "extra": "mean: 4.616028785999982 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "13fdda268cf349fe5b3c328a8004cd6de67efa40",
          "message": "Fix the azure logging handler in Everest\n\n* Fix the azure logging handler in Everest\r\n\r\nCo-authored-by: DanSava <dan.sava42@gmail.com>",
          "timestamp": "2024-12-03T17:21:47+02:00",
          "tree_id": "42d983a6caf0663e408a354e283e628097c4dc9d",
          "url": "https://github.com/equinor/ert/commit/13fdda268cf349fe5b3c328a8004cd6de67efa40"
        },
        "date": 1733239420814,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21449748614004083,
            "unit": "iter/sec",
            "range": "stddev: 0.03604122720208032",
            "extra": "mean: 4.662059299600003 sec\nrounds: 5"
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
          "id": "831e34bdc8f008ba477059001b77c014bbd4dcac",
          "message": "Add more DATA parsing tests for num_cpu",
          "timestamp": "2024-12-04T09:50:16+01:00",
          "tree_id": "d1d3dd485e25ceb50a309471ed11c66920eca776",
          "url": "https://github.com/equinor/ert/commit/831e34bdc8f008ba477059001b77c014bbd4dcac"
        },
        "date": 1733302327211,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21041206438626736,
            "unit": "iter/sec",
            "range": "stddev: 0.030678677762065683",
            "extra": "mean: 4.7525791970000055 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "11fbe8046a071ecac07efb99f134776ee039178d",
          "message": "Fix everest doc test failing due to missing coverage",
          "timestamp": "2024-12-04T09:58:55+01:00",
          "tree_id": "b4f645cd904b06e00c999d1e9461bcc453495bd8",
          "url": "https://github.com/equinor/ert/commit/11fbe8046a071ecac07efb99f134776ee039178d"
        },
        "date": 1733302843052,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21174496168005535,
            "unit": "iter/sec",
            "range": "stddev: 0.09191409000932561",
            "extra": "mean: 4.722662546800007 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "dcf56325e626f4c5c6fe101674e274fa27ef23d3",
          "message": "Fix failing everest test due to doc not generating cov",
          "timestamp": "2024-12-04T10:45:02+01:00",
          "tree_id": "aa76c2fbffbc5342f9fe4793753a74c120ccf1a5",
          "url": "https://github.com/equinor/ert/commit/dcf56325e626f4c5c6fe101674e274fa27ef23d3"
        },
        "date": 1733305610478,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21572534969006749,
            "unit": "iter/sec",
            "range": "stddev: 0.030114381515328442",
            "extra": "mean: 4.63552383359999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "0ba06268022b87fa46649581e9a5f3bd5b5b2c19",
          "message": "Remove backport of datetime functions\n\nThis is no longer needed as python <3.11 is not supported anymore",
          "timestamp": "2024-12-04T13:02:52+01:00",
          "tree_id": "be197d4e9a45e95c6d5544842c721de218fa4b7b",
          "url": "https://github.com/equinor/ert/commit/0ba06268022b87fa46649581e9a5f3bd5b5b2c19"
        },
        "date": 1733313900339,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21409340979839384,
            "unit": "iter/sec",
            "range": "stddev: 0.042338890828999935",
            "extra": "mean: 4.670858392800011 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "andrli@equinor.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "committer": {
            "email": "60844986+larsevj@users.noreply.github.com",
            "name": "Lars Evje",
            "username": "larsevj"
          },
          "distinct": true,
          "id": "471df83cff13d972c9e91db27118d222f062da91",
          "message": "Update run_dialog fm_label test to verify item clicked",
          "timestamp": "2024-12-05T09:51:05+01:00",
          "tree_id": "94d30d79b7c694c047f13c41822ea6f1c72e62c5",
          "url": "https://github.com/equinor/ert/commit/471df83cff13d972c9e91db27118d222f062da91"
        },
        "date": 1733388774066,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2153187352691876,
            "unit": "iter/sec",
            "range": "stddev: 0.01770439466924137",
            "extra": "mean: 4.644277697200005 sec\nrounds: 5"
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
          "id": "6c86e6aeb4e0610945156e6a845f145e0227f296",
          "message": "Move shell jobs test",
          "timestamp": "2024-12-05T10:33:11+01:00",
          "tree_id": "0914236da2ed50bd3df1a6a90410f1513e99fb30",
          "url": "https://github.com/equinor/ert/commit/6c86e6aeb4e0610945156e6a845f145e0227f296"
        },
        "date": 1733391302229,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21364418028765084,
            "unit": "iter/sec",
            "range": "stddev: 0.031307127655364836",
            "extra": "mean: 4.6806798044 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "a144f88a5ee5a029b2f906714f1d1e1f8f33c761",
          "message": "Use sampled_from instead of parametrize in test\n\nThis is in order to speed up the test by sampling 100 times\ninsted of 25*100 times.",
          "timestamp": "2024-12-05T10:59:12+01:00",
          "tree_id": "65b40521481f024f47a778814690989590970395",
          "url": "https://github.com/equinor/ert/commit/a144f88a5ee5a029b2f906714f1d1e1f8f33c761"
        },
        "date": 1733392860206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2153005849111255,
            "unit": "iter/sec",
            "range": "stddev: 0.033963081775852486",
            "extra": "mean: 4.644669220999992 sec\nrounds: 5"
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
            "email": "107626001+jonathan-eq@users.noreply.github.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "distinct": true,
          "id": "4e31e0bec0610e3f4a32014e9e3fdf749a8a915f",
          "message": "Fix progress bar not updating realization count for new iterations\n\nThis commit fixes the bug introduced in 31e607b066ab79415671f83f2d57c7400c4d4e98, where the status reporting in GUI was done the same way when rerunning failed realizations, and running new iterations. This is an issue because when rerunning failed realizations, we want to show all realizations and add the finished/failed count from the previous run, while new iterations should drop the failed realizations altogether.",
          "timestamp": "2024-12-05T14:57:40+01:00",
          "tree_id": "6b089c914f3f221b7f1bc023a0d56d356a2c38ed",
          "url": "https://github.com/equinor/ert/commit/4e31e0bec0610e3f4a32014e9e3fdf749a8a915f"
        },
        "date": 1733407175503,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21154862888533055,
            "unit": "iter/sec",
            "range": "stddev: 0.03076221655735647",
            "extra": "mean: 4.72704552739998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "434b77391a68481e0cc6c5584b43a9e125866729",
          "message": "Fix crash when reading num_cpu from non-ascii files",
          "timestamp": "2024-12-05T15:10:31+01:00",
          "tree_id": "2768c7cd0e2548e17b2772378334c9aed4ac23a4",
          "url": "https://github.com/equinor/ert/commit/434b77391a68481e0cc6c5584b43a9e125866729"
        },
        "date": 1733407941702,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2167327356963286,
            "unit": "iter/sec",
            "range": "stddev: 0.025162739190932745",
            "extra": "mean: 4.6139776568 sec\nrounds: 5"
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
          "id": "b903bb03be7e401622f604442691b36e5addbde4",
          "message": "Remove overriding of tempfile.tempdir\n\nWhen the jobs are executed on the cluster, /user/run/<userid> is not set up,\nthough XDG_RUNTIME_DIR points to it. This is not a problem for ert as it\nruns the main application locally, but is a problem for Everest where the\nmain application runs on the cluster. So the way lsf logs in to the node is the reason.",
          "timestamp": "2024-12-05T21:13:16+01:00",
          "tree_id": "656ff4f7053e3f065b481da16cf0864a80775c79",
          "url": "https://github.com/equinor/ert/commit/b903bb03be7e401622f604442691b36e5addbde4"
        },
        "date": 1733429707584,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21689946115874717,
            "unit": "iter/sec",
            "range": "stddev: 0.04384166878033591",
            "extra": "mean: 4.610431001799986 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "03903bbc50fbd911313469a10d29c4d43636917c",
          "message": "Improve exit code detection in the slurm driver (#9440)\n\nImprove exit code detection in the slurm driver",
          "timestamp": "2024-12-06T10:27:57+01:00",
          "tree_id": "e21e8c5f40dffa7bee9d0754cc27584a9846b319",
          "url": "https://github.com/equinor/ert/commit/03903bbc50fbd911313469a10d29c4d43636917c"
        },
        "date": 1733477386342,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21656208453872952,
            "unit": "iter/sec",
            "range": "stddev: 0.028287705830547526",
            "extra": "mean: 4.6176134761999945 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "oyveid@equinor.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "committer": {
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "aa45608cf666c28b38c32d6b339fe9ee04cf6cb9",
          "message": "Move addition of activate script",
          "timestamp": "2024-12-06T10:38:26+01:00",
          "tree_id": "0e5558d1ba63b8c98d1191b1779ffa407a4459f3",
          "url": "https://github.com/equinor/ert/commit/aa45608cf666c28b38c32d6b339fe9ee04cf6cb9"
        },
        "date": 1733478014035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21382553020849498,
            "unit": "iter/sec",
            "range": "stddev: 0.05113461513636054",
            "extra": "mean: 4.67671002159999 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}