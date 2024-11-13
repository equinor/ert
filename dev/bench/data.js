window.BENCHMARK_DATA = {
  "lastUpdate": 1731490189489,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "66aa073a0831991022c17dff7a5fd09688298e37",
          "message": "Use maxprocesses=2 for cli tests\n\nCo-authored-by: Eivind Jahren <ejah@equinor.com>",
          "timestamp": "2024-11-01T11:32:36+01:00",
          "tree_id": "2262601ceb4df8b82c20f24d137caf1c9f2e402a",
          "url": "https://github.com/equinor/ert/commit/66aa073a0831991022c17dff7a5fd09688298e37"
        },
        "date": 1730457263680,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19656487553447052,
            "unit": "iter/sec",
            "range": "stddev: 0.03869466312127469",
            "extra": "mean: 5.087378898599996 sec\nrounds: 5"
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
          "id": "044db5aec48bbda9839cf197c7faac5c8e72a058",
          "message": "Revert \"Increase sleep in memory profile test from 0.1 -> 0.15\"\n\nThis reverts commit 98576cc8ad12e751a19340e8fa350ef08ed3ee59.\n\nChanging the sleep time affects the rate of memory allocation,\nwhich the assert further down depends on.",
          "timestamp": "2024-11-01T12:39:24+01:00",
          "tree_id": "b13dbc3fab2b8f853aa8b692b223d51e71a795bc",
          "url": "https://github.com/equinor/ert/commit/044db5aec48bbda9839cf197c7faac5c8e72a058"
        },
        "date": 1730461281432,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19200743007680682,
            "unit": "iter/sec",
            "range": "stddev: 0.04340572702924938",
            "extra": "mean: 5.2081317873999975 sec\nrounds: 5"
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
          "id": "7ee89c18b36844c0e4c3ba0f886ecccdc1d7e552",
          "message": "Mitigate flakiness in memory profiling\n\nAnd add some explanation for further debugging",
          "timestamp": "2024-11-01T14:21:48+01:00",
          "tree_id": "99e7dbc9660f1203352392ac251f9504a719f21a",
          "url": "https://github.com/equinor/ert/commit/7ee89c18b36844c0e4c3ba0f886ecccdc1d7e552"
        },
        "date": 1730467418469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19083839842672864,
            "unit": "iter/sec",
            "range": "stddev: 0.034027747519319154",
            "extra": "mean: 5.2400355916000025 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49289030+HakonSohoel@users.noreply.github.com",
            "name": "Håkon Steinkopf Søhoel",
            "username": "HakonSohoel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c34060ca08b824c1e5dbedd21cca552dff98b796",
          "message": "Improve logging with open telemetry traces (#9083)\n\nAdd a span processor through the add_span_processor pluggin hook\r\nto export trace information to e.g. azure\r\n---------\r\n\r\nCo-authored-by: Andreas Eknes Lie <andrli@equinor.com>",
          "timestamp": "2024-11-01T14:25:59+01:00",
          "tree_id": "a4b2c2ab5e5967b8505c052d41534d985e8bde95",
          "url": "https://github.com/equinor/ert/commit/c34060ca08b824c1e5dbedd21cca552dff98b796"
        },
        "date": 1730467673141,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1939554861378494,
            "unit": "iter/sec",
            "range": "stddev: 0.02633452409202933",
            "extra": "mean: 5.1558221935999935 sec\nrounds: 5"
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
          "id": "87136c7cbf2f966dced604e36a2dde82a129285d",
          "message": "Add timeout for server",
          "timestamp": "2024-11-01T14:59:14+01:00",
          "tree_id": "095dc0a0f2a245530d269197313acb2f998ecec2",
          "url": "https://github.com/equinor/ert/commit/87136c7cbf2f966dced604e36a2dde82a129285d"
        },
        "date": 1730469665407,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1898364121527562,
            "unit": "iter/sec",
            "range": "stddev: 0.03155378385903209",
            "extra": "mean: 5.267693318999977 sec\nrounds: 5"
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
          "id": "1488d88244f4b2bb2cd165a7197fd0846c1b701a",
          "message": "Remove ert_config from batch sim",
          "timestamp": "2024-11-01T15:03:13+01:00",
          "tree_id": "0806b0089def2c042be2389c9d12d06ee60aaae7",
          "url": "https://github.com/equinor/ert/commit/1488d88244f4b2bb2cd165a7197fd0846c1b701a"
        },
        "date": 1730469902449,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1926733180812877,
            "unit": "iter/sec",
            "range": "stddev: 0.017998264994396853",
            "extra": "mean: 5.190132240199995 sec\nrounds: 5"
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
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "90d11ec9604f2d3b4b2a12cd90ba19ec58fbf1ca",
          "message": "Make scheduler execute yield during spawning of realizations\n\nStarting the realizations in scheduler was blocking all other async tasks\nfrom running. Nothing could connect to ensemble evaluator during this.\nUnder heavy load this could cause Monitor to time out and fail. Now we will\nsleep(0) between each time we create a new subprocess. This will allow\nother asyncio tasks to run.",
          "timestamp": "2024-11-01T15:15:27+01:00",
          "tree_id": "5fd69702e1f912ba4c614ab240104c1f7b93c28b",
          "url": "https://github.com/equinor/ert/commit/90d11ec9604f2d3b4b2a12cd90ba19ec58fbf1ca"
        },
        "date": 1730470638886,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19423774875696875,
            "unit": "iter/sec",
            "range": "stddev: 0.035257273320927464",
            "extra": "mean: 5.148329850400012 sec\nrounds: 5"
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
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "distinct": true,
          "id": "8c679c1f3753359fdf656db36faa5721e182a865",
          "message": "Update ropt dependency to 0.9",
          "timestamp": "2024-11-04T11:10:22+01:00",
          "tree_id": "213486ade147c9856172d5a8c2d4b11b20381248",
          "url": "https://github.com/equinor/ert/commit/8c679c1f3753359fdf656db36faa5721e182a865"
        },
        "date": 1730715132812,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19240231102079516,
            "unit": "iter/sec",
            "range": "stddev: 0.028166166099757847",
            "extra": "mean: 5.1974427682000055 sec\nrounds: 5"
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
          "id": "05dfe1711c20226ee73ff22eb3794272b6b794e7",
          "message": "Create driver from QueueOptions instead of QueueConfig",
          "timestamp": "2024-11-05T10:21:14+01:00",
          "tree_id": "a2240fbc1f1acd9b98b2dff2fd72c14acf3c9bc6",
          "url": "https://github.com/equinor/ert/commit/05dfe1711c20226ee73ff22eb3794272b6b794e7"
        },
        "date": 1730798583967,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19657331009998202,
            "unit": "iter/sec",
            "range": "stddev: 0.04104420423443821",
            "extra": "mean: 5.0871606094000015 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e310860625e56f8daad713991ef56f35ef04d400",
          "message": "Transpose before calculating covariance in test\n\nnp.cov expects rows to be parameters and columns to be realizations",
          "timestamp": "2024-11-05T10:24:16+01:00",
          "tree_id": "01c0b3041f0abd978231c71184edbf4749b148c9",
          "url": "https://github.com/equinor/ert/commit/e310860625e56f8daad713991ef56f35ef04d400"
        },
        "date": 1730798770057,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19046176259498293,
            "unit": "iter/sec",
            "range": "stddev: 0.07114044650732476",
            "extra": "mean: 5.250397698600011 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "stephan.dehoop@tno.nl",
            "name": "Stephan de Hoop",
            "username": "StephanDeHoop"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1f00a17aa7bce09e93740ffdb3754285abaacbe5",
          "message": "Fix small typo in Everest docs (#9153)\n\nFix small typo in docs",
          "timestamp": "2024-11-05T10:41:15+01:00",
          "tree_id": "c63a7abd0552cfa75a94ce3d4bfc4ec2cd0c5273",
          "url": "https://github.com/equinor/ert/commit/1f00a17aa7bce09e93740ffdb3754285abaacbe5"
        },
        "date": 1730799781183,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19694008109762362,
            "unit": "iter/sec",
            "range": "stddev: 0.02300224569439814",
            "extra": "mean: 5.077686545199998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49289030+HakonSohoel@users.noreply.github.com",
            "name": "Håkon Steinkopf Søhoel",
            "username": "HakonSohoel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59453cf198848c74e942eb0c3a4fb34d34c54c21",
          "message": "Fix pytest option typo (#9141)",
          "timestamp": "2024-11-05T12:19:47+01:00",
          "tree_id": "aa4ba7c44b9a629e2f1cc9656816b08e7a0923bf",
          "url": "https://github.com/equinor/ert/commit/59453cf198848c74e942eb0c3a4fb34d34c54c21"
        },
        "date": 1730805702469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19317858907262322,
            "unit": "iter/sec",
            "range": "stddev: 0.010193189508555507",
            "extra": "mean: 5.176557116400005 sec\nrounds: 5"
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
          "id": "4b18368cbac651d45bd0e30845686e3efd171810",
          "message": "Remove simulation server option and add deprecation warning is present in config",
          "timestamp": "2024-11-06T04:03:35+09:00",
          "tree_id": "e4299ca59a7a4d9af15821a802d6ef27ff856179",
          "url": "https://github.com/equinor/ert/commit/4b18368cbac651d45bd0e30845686e3efd171810"
        },
        "date": 1730833536856,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19389436321681078,
            "unit": "iter/sec",
            "range": "stddev: 0.06516216173492596",
            "extra": "mean: 5.157447506000006 sec\nrounds: 5"
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
          "id": "721f3c6d24038a0a55677cd81ef169af87a81d95",
          "message": "Make simulator fm invocation more explicit",
          "timestamp": "2024-11-06T09:17:29+01:00",
          "tree_id": "09af707c0a6520b44b2ee84198f5f546ebd9428d",
          "url": "https://github.com/equinor/ert/commit/721f3c6d24038a0a55677cd81ef169af87a81d95"
        },
        "date": 1730881163707,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1924269346460438,
            "unit": "iter/sec",
            "range": "stddev: 0.0370351370942405",
            "extra": "mean: 5.196777685200004 sec\nrounds: 5"
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
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "distinct": true,
          "id": "0c6f980d6f782b7619826bf2f4e74b8ae5e12f1b",
          "message": "Remove the get_forward_models hook",
          "timestamp": "2024-11-08T12:28:23+01:00",
          "tree_id": "4cb0b165ad455bb5073aeaf946a52ed7b8fd166f",
          "url": "https://github.com/equinor/ert/commit/0c6f980d6f782b7619826bf2f4e74b8ae5e12f1b"
        },
        "date": 1731065415107,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19112839944377166,
            "unit": "iter/sec",
            "range": "stddev: 0.03579840874600964",
            "extra": "mean: 5.232084833599998 sec\nrounds: 5"
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
          "id": "1b29b3a37efd53b56fe86ed181069306a4b3b5b3",
          "message": "Enable load results manually from any available iteration\n\n* Allow non-zero iteration number when creating experiment-ensemble pair using CreateExperimentDialog\r\n\r\n* Enable load results manually from any available iteration",
          "timestamp": "2024-11-11T11:23:05+02:00",
          "tree_id": "8e02d3e23a4fe3fb59917ec29893d77460f9379f",
          "url": "https://github.com/equinor/ert/commit/1b29b3a37efd53b56fe86ed181069306a4b3b5b3"
        },
        "date": 1731317093034,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19611852843634342,
            "unit": "iter/sec",
            "range": "stddev: 0.024245559668833395",
            "extra": "mean: 5.098957288600002 sec\nrounds: 5"
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
            "email": "107626001+jonathan-eq@users.noreply.github.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "distinct": true,
          "id": "a7559fdcd2bf70dcf59e9c14c9cc97c00d76cf5b",
          "message": "Freeze websockets at < 14\n\nNew version is incompatible with our current code",
          "timestamp": "2024-11-11T15:44:22+01:00",
          "tree_id": "64eb439aa57f476e8f43bf6e568ae30cd0d13790",
          "url": "https://github.com/equinor/ert/commit/a7559fdcd2bf70dcf59e9c14c9cc97c00d76cf5b"
        },
        "date": 1731336371907,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19157688324906472,
            "unit": "iter/sec",
            "range": "stddev: 0.03577791452492691",
            "extra": "mean: 5.219836459599998 sec\nrounds: 5"
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
          "id": "3fec61fe1e32d284424476a47713a662fdf04fec",
          "message": "Disable modes that do parameter updates if there are no params to update\n\nIf all parameters have UPDATE:FALSE set, then we should not let the user\nselect any mode that does parameter updates.",
          "timestamp": "2024-11-11T20:44:48+01:00",
          "tree_id": "1df8258edfecb6486268ac9a7943afa6723a490f",
          "url": "https://github.com/equinor/ert/commit/3fec61fe1e32d284424476a47713a662fdf04fec"
        },
        "date": 1731354399209,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19687312447796668,
            "unit": "iter/sec",
            "range": "stddev: 0.02858994384048197",
            "extra": "mean: 5.079413468200005 sec\nrounds: 5"
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
          "id": "08b672c8498636fdb1da41a5c8380f54ff481ade",
          "message": "Mitigate flakyness on busy test nodes\n\nA kill window of 1 second is not enough on real-life test nodes.",
          "timestamp": "2024-11-12T09:15:54+01:00",
          "tree_id": "e436340dbec313ed44706a92af0f3e3f3a2c264e",
          "url": "https://github.com/equinor/ert/commit/08b672c8498636fdb1da41a5c8380f54ff481ade"
        },
        "date": 1731399468448,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19543837210645734,
            "unit": "iter/sec",
            "range": "stddev: 0.04032319220786674",
            "extra": "mean: 5.11670246340002 sec\nrounds: 5"
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
          "id": "44ece921ec48834ffddb256fdc93f9e9561c7020",
          "message": "Update pre-commit: ruff v0.6.9 -> v0.7.3",
          "timestamp": "2024-11-12T09:49:30+01:00",
          "tree_id": "f361723a24ce7be08bf6531727e7ddfd132a0a2c",
          "url": "https://github.com/equinor/ert/commit/44ece921ec48834ffddb256fdc93f9e9561c7020"
        },
        "date": 1731401496656,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1902289437654308,
            "unit": "iter/sec",
            "range": "stddev: 0.030989492307420905",
            "extra": "mean: 5.2568235948000055 sec\nrounds: 5"
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
          "id": "b5d36717b7bdc15879f02c8a44ae614dea95a7eb",
          "message": "Add default values using Pandas assign in design_matrix",
          "timestamp": "2024-11-12T11:12:31+01:00",
          "tree_id": "5f970c12fb5113a4e6cf6080b1d79de5a2314a8f",
          "url": "https://github.com/equinor/ert/commit/b5d36717b7bdc15879f02c8a44ae614dea95a7eb"
        },
        "date": 1731406463776,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18967021305501305,
            "unit": "iter/sec",
            "range": "stddev: 0.054413618347240345",
            "extra": "mean: 5.272309151200005 sec\nrounds: 5"
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
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "5b96e8a9a4cfcd809941596212ea637dcd493018",
          "message": "Add just command helper tool to repository",
          "timestamp": "2024-11-12T12:22:36+01:00",
          "tree_id": "ba4d5c9c3463b1b6dfa076ff25a6dd63bdc30150",
          "url": "https://github.com/equinor/ert/commit/5b96e8a9a4cfcd809941596212ea637dcd493018"
        },
        "date": 1731410677520,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.190592177302434,
            "unit": "iter/sec",
            "range": "stddev: 0.015545395168153927",
            "extra": "mean: 5.246805058600006 sec\nrounds: 5"
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
          "id": "f088b4b9140945a2effab4cfa38a42c4aac0c6c0",
          "message": "Remove macos fail flag export tests",
          "timestamp": "2024-11-12T14:36:03+01:00",
          "tree_id": "fdd56e044dfabfb8d111a9a721ad51c38566b218",
          "url": "https://github.com/equinor/ert/commit/f088b4b9140945a2effab4cfa38a42c4aac0c6c0"
        },
        "date": 1731418677331,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.195301811719276,
            "unit": "iter/sec",
            "range": "stddev: 0.008535186557729876",
            "extra": "mean: 5.120280202199996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yngve-sk@users.noreply.github.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fecb69209de6b4751f75007404cf0ac04555e464",
          "message": "Add snapshot test for everest API",
          "timestamp": "2024-11-12T13:50:20Z",
          "tree_id": "229c71a030ca1bcc6d6b7299e6b372efc66ecbca",
          "url": "https://github.com/equinor/ert/commit/fecb69209de6b4751f75007404cf0ac04555e464"
        },
        "date": 1731419533395,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19151137827394382,
            "unit": "iter/sec",
            "range": "stddev: 0.02492280900488543",
            "extra": "mean: 5.221621864000002 sec\nrounds: 5"
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
          "id": "e50f2d3463133db536e3204cd7c27b528122a4e7",
          "message": "Assume non-LSF host error is flaky\n\nThe LSF driver experiences crashes stemming from bsub returning with the\nerror message 'Request from non-LSF host rejected'. There are reasons to\nbelieve this is not a permanent error, but some flakyness in the IP\ninfrastructure, and thus should should be categorized as a retriable\nfailure.\n\nThe reason for believing this is flakyness is mostly from the fact that\nthe same error is also seen on 'bjobs'-calls. If it was a permanent\nfailure scenario, there would be an enourmous amount of error from these\nbjobs calls, but there is not.",
          "timestamp": "2024-11-12T14:55:36+01:00",
          "tree_id": "bf29e38b8c6c51bab62bc55819496f2308075b34",
          "url": "https://github.com/equinor/ert/commit/e50f2d3463133db536e3204cd7c27b528122a4e7"
        },
        "date": 1731419847936,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1855084750782697,
            "unit": "iter/sec",
            "range": "stddev: 0.12035984176026342",
            "extra": "mean: 5.3905892956 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49289030+HakonSohoel@users.noreply.github.com",
            "name": "Håkon Steinkopf Søhoel",
            "username": "HakonSohoel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "efb812cf823994775f4ddea84a3c19b889acf9dd",
          "message": "Add trace ID to clipboard debug info and title bar (#9157)",
          "timestamp": "2024-11-12T15:05:33+01:00",
          "tree_id": "be794f114365a4a6196b469bd1ad04134adc2c69",
          "url": "https://github.com/equinor/ert/commit/efb812cf823994775f4ddea84a3c19b889acf9dd"
        },
        "date": 1731420442192,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19307666222800574,
            "unit": "iter/sec",
            "range": "stddev: 0.01839081835385295",
            "extra": "mean: 5.179289865800001 sec\nrounds: 5"
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
          "id": "22b7f7142ca8232fb9c5f39fbfb99cfc58aca5fa",
          "message": "Fixup ert plugin documentation\n\n- Specifies package structure assumption\n- Adds pyproject.toml example\n- Fixes syntax error in code example",
          "timestamp": "2024-11-12T15:51:10+01:00",
          "tree_id": "cf7a53afb52243068701175efdb21ae6700ff4bc",
          "url": "https://github.com/equinor/ert/commit/22b7f7142ca8232fb9c5f39fbfb99cfc58aca5fa"
        },
        "date": 1731423238294,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1927639173767591,
            "unit": "iter/sec",
            "range": "stddev: 0.0368780184250326",
            "extra": "mean: 5.187692871200005 sec\nrounds: 5"
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
          "id": "93081a13cfba2dd439943a5a72912d786e3e2890",
          "message": "Move everserver config to ServerConfig",
          "timestamp": "2024-11-13T07:48:44+01:00",
          "tree_id": "a1cc3c08e11020bf6233519842e48d7e4c985a60",
          "url": "https://github.com/equinor/ert/commit/93081a13cfba2dd439943a5a72912d786e3e2890"
        },
        "date": 1731480640195,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19134390136318888,
            "unit": "iter/sec",
            "range": "stddev: 0.01955277528329969",
            "extra": "mean: 5.226192174799996 sec\nrounds: 5"
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
          "id": "03cfa25311451e196ded467b2e260b41a79d6587",
          "message": "Remove old job queue test snapshot",
          "timestamp": "2024-11-13T07:59:48+01:00",
          "tree_id": "932ccdc696aefc5c0347d0d0f614718ebec01a35",
          "url": "https://github.com/equinor/ert/commit/03cfa25311451e196ded467b2e260b41a79d6587"
        },
        "date": 1731481306024,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19111658429816888,
            "unit": "iter/sec",
            "range": "stddev: 0.030203551263009",
            "extra": "mean: 5.232408289800003 sec\nrounds: 5"
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
          "id": "5ac831ed67f2d15393bdc9fa2da53a5014b5e3e2",
          "message": "Add warning when everest-models file outputs do not match everest objectives",
          "timestamp": "2024-11-13T18:27:54+09:00",
          "tree_id": "96be8224578478c256319794909450a7966210c4",
          "url": "https://github.com/equinor/ert/commit/5ac831ed67f2d15393bdc9fa2da53a5014b5e3e2"
        },
        "date": 1731490188870,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19508941088438342,
            "unit": "iter/sec",
            "range": "stddev: 0.02441490217600472",
            "extra": "mean: 5.1258548347999975 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}