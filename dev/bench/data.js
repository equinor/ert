window.BENCHMARK_DATA = {
  "lastUpdate": 1733749734529,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "5061a9426cc6f5ae2db17d2f958a6c9ecfe0cb58",
          "message": "Inline redefinition of components in test",
          "timestamp": "2024-12-06T12:34:23+01:00",
          "tree_id": "e44e579d3509cb6d869a7ded31288311d70c227f",
          "url": "https://github.com/equinor/ert/commit/5061a9426cc6f5ae2db17d2f958a6c9ecfe0cb58"
        },
        "date": 1733484970869,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21343338675248258,
            "unit": "iter/sec",
            "range": "stddev: 0.059976100777467865",
            "extra": "mean: 4.685302591200008 sec\nrounds: 5"
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
          "id": "710e6d5d13a16ec1df4340910b52ca6328f3b6bb",
          "message": "Migrate finalized keys for response configs",
          "timestamp": "2024-12-06T12:34:42+01:00",
          "tree_id": "d5c0168545e3cc674dbc6545b05a0c8189767d37",
          "url": "https://github.com/equinor/ert/commit/710e6d5d13a16ec1df4340910b52ca6328f3b6bb"
        },
        "date": 1733484987413,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21560824691593727,
            "unit": "iter/sec",
            "range": "stddev: 0.029463112519569366",
            "extra": "mean: 4.638041514199995 sec\nrounds: 5"
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
          "id": "4b34fc94ceab52c8ad34e5059837b7ec26e5dcb9",
          "message": "Delete unused `evaluate` method in ensemble_evaluator_utils",
          "timestamp": "2024-12-06T12:34:46+01:00",
          "tree_id": "3f250a19d7c5ac087dd71f3e56a1c1141feeaa7a",
          "url": "https://github.com/equinor/ert/commit/4b34fc94ceab52c8ad34e5059837b7ec26e5dcb9"
        },
        "date": 1733484995231,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.208689645651837,
            "unit": "iter/sec",
            "range": "stddev: 0.05284187912160273",
            "extra": "mean: 4.791804580799993 sec\nrounds: 5"
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
          "id": "726a273e44984d59214426751203284b10d27efe",
          "message": "Have ModelConfig output more noticable warning when malformatted runpath\n\nThis commit makes ModelConfig emit a ConfigWarning if the input runpath does not contain `<ITER>` or `<IENS>`. This was previously only a warning in the logs, but it should be more noticable.",
          "timestamp": "2024-12-06T12:50:39+01:00",
          "tree_id": "9a6f2404d6a597848fafec7c30f82ba0d17c5fe1",
          "url": "https://github.com/equinor/ert/commit/726a273e44984d59214426751203284b10d27efe"
        },
        "date": 1733485947918,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21179165591509186,
            "unit": "iter/sec",
            "range": "stddev: 0.06572133260757662",
            "extra": "mean: 4.7216213295999925 sec\nrounds: 5"
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
          "id": "90c14a927fb0e4c1de9e08a88f7599f329342a0a",
          "message": "Improve UX for permission errors in storage\n\nThis commit:\n* Improves the error message displayed when the dark storage server does not have access to the storage path.\n* Makes the dark storage server return a response with status code 401 - unauthorized when the `get_ensemble_record` endpoint fails due to `PermissionError`.\n* Makes the failed message in `LegacyEnsemble._evaluate_inner` omit stacktrace when it failed due to PermissionError, making it shorter and more consise.",
          "timestamp": "2024-12-06T12:51:10+01:00",
          "tree_id": "5c8c3d177362548299096fd3e12b750c461d7570",
          "url": "https://github.com/equinor/ert/commit/90c14a927fb0e4c1de9e08a88f7599f329342a0a"
        },
        "date": 1733485981654,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2137260034404691,
            "unit": "iter/sec",
            "range": "stddev: 0.05363075399486037",
            "extra": "mean: 4.678887846599997 sec\nrounds: 5"
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
            "email": "frodeaarstad@gmail.com",
            "name": "Frode Aarstad",
            "username": "frode-aarstad"
          },
          "distinct": true,
          "id": "baeb4f52bbbf45a292b0ca943d700c4d485c6c5b",
          "message": "Update everest snapshot egg-py311.csv",
          "timestamp": "2024-12-06T18:42:55+01:00",
          "tree_id": "586c7b283858f837abcdc0c627535b72d1accfed",
          "url": "https://github.com/equinor/ert/commit/baeb4f52bbbf45a292b0ca943d700c4d485c6c5b"
        },
        "date": 1733507089070,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21517625560161308,
            "unit": "iter/sec",
            "range": "stddev: 0.07914358763160681",
            "extra": "mean: 4.647352921000004 sec\nrounds: 5"
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
          "id": "a5bfbdb3e14fd6eaf0155dc545fdcb8c1ba9b2d3",
          "message": "Mute marginal cpu overspending\n\nThere exists logs that a user has overspent with a factor of 1.0. This is not very\ninteresting, so skip logging anything that we don't find significant.",
          "timestamp": "2024-12-09T12:54:57+01:00",
          "tree_id": "531f7b1e539f789dc17711ef32c2ede1cad32ff9",
          "url": "https://github.com/equinor/ert/commit/a5bfbdb3e14fd6eaf0155dc545fdcb8c1ba9b2d3"
        },
        "date": 1733745408650,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22054059747100946,
            "unit": "iter/sec",
            "range": "stddev: 0.021143488419735627",
            "extra": "mean: 4.534312555000002 sec\nrounds: 5"
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
          "id": "ea78bd8890343d1c99337be738c3ccae8c32fd45",
          "message": "Only set active realizations from selected ensemble when restart is checked es_mda",
          "timestamp": "2024-12-09T14:06:59+01:00",
          "tree_id": "07962ecd11797d4dc9a46e969fda438d1acbeead",
          "url": "https://github.com/equinor/ert/commit/ea78bd8890343d1c99337be738c3ccae8c32fd45"
        },
        "date": 1733749733640,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21473519569661298,
            "unit": "iter/sec",
            "range": "stddev: 0.09679680966034868",
            "extra": "mean: 4.656898449999983 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}