window.BENCHMARK_DATA = {
  "lastUpdate": 1735906397897,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "01156bb8355596d8dcb22570d963091bdca71c74",
          "message": "Add specific exceptions to bad-dunder-names",
          "timestamp": "2024-12-18T09:04:48+01:00",
          "tree_id": "b5a5857550018617a9288eed2e4d5d33898fe280",
          "url": "https://github.com/equinor/ert/commit/01156bb8355596d8dcb22570d963091bdca71c74"
        },
        "date": 1734509233256,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21463586830074877,
            "unit": "iter/sec",
            "range": "stddev: 0.010339623610527466",
            "extra": "mean: 4.6590535306 sec\nrounds: 5"
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
          "id": "a6649a18d307b353de1e01428b189a6b335108e6",
          "message": "Unpin pydantic",
          "timestamp": "2024-12-18T13:49:34+01:00",
          "tree_id": "0b3fc34b704dca4f183d131375f05b1033bb35b8",
          "url": "https://github.com/equinor/ert/commit/a6649a18d307b353de1e01428b189a6b335108e6"
        },
        "date": 1734526298593,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2171199435754667,
            "unit": "iter/sec",
            "range": "stddev: 0.04954297387413186",
            "extra": "mean: 4.6057491703999975 sec\nrounds: 5"
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
          "id": "a0dc76e4dadb0d08bfc69afc2adbec7137306172",
          "message": "Replace deprecated polars count with len",
          "timestamp": "2024-12-18T14:22:54+01:00",
          "tree_id": "6316496331e025e9011e73ed857da1f77ba339bb",
          "url": "https://github.com/equinor/ert/commit/a0dc76e4dadb0d08bfc69afc2adbec7137306172"
        },
        "date": 1734528293991,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.20917190507829692,
            "unit": "iter/sec",
            "range": "stddev: 0.03708193137887556",
            "extra": "mean: 4.780756763800002 sec\nrounds: 5"
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
          "id": "b7b57747012bfc9ac1e4c8f611d56f9f6879055d",
          "message": "Remove unused test data",
          "timestamp": "2024-12-18T14:36:21+01:00",
          "tree_id": "a4fa814c30272156a7523083cffbd249f9aab56d",
          "url": "https://github.com/equinor/ert/commit/b7b57747012bfc9ac1e4c8f611d56f9f6879055d"
        },
        "date": 1734529092500,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21827420559198069,
            "unit": "iter/sec",
            "range": "stddev: 0.033091739166018826",
            "extra": "mean: 4.581393377599994 sec\nrounds: 5"
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
          "id": "7f72c056b7b55de316903d2dc4a60e52954be9b0",
          "message": "Emit signal when tab changes to select realization\n\nCo-authored-by: Jonathan Karlsen <jonak@equinor.com>",
          "timestamp": "2024-12-18T14:56:29+01:00",
          "tree_id": "8f85eba314f6e8eb21b8ff6e5c9e3693a86c7e37",
          "url": "https://github.com/equinor/ert/commit/7f72c056b7b55de316903d2dc4a60e52954be9b0"
        },
        "date": 1734530305624,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21693890847473488,
            "unit": "iter/sec",
            "range": "stddev: 0.03951958364486",
            "extra": "mean: 4.609592659200007 sec\nrounds: 5"
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
          "id": "6ce4714e1c65b04473fb1cd0f74e1e951fc39f28",
          "message": "Solve ruff literal-membership\n\nruff check --fix --unsafe-fixes",
          "timestamp": "2024-12-18T21:45:13+01:00",
          "tree_id": "81d4bd24f8e000c06c0c1d426f8840e8bea028a1",
          "url": "https://github.com/equinor/ert/commit/6ce4714e1c65b04473fb1cd0f74e1e951fc39f28"
        },
        "date": 1734554827693,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2188614973114114,
            "unit": "iter/sec",
            "range": "stddev: 0.019665328952347745",
            "extra": "mean: 4.569099692199996 sec\nrounds: 5"
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
          "id": "a95d0d2f2bd7bb6963d272c0f56635e0a47ed298",
          "message": "Rename job_dispatch -> fm_dispatch",
          "timestamp": "2024-12-19T07:40:58+01:00",
          "tree_id": "62eb4933ad6ae699b1df161e110c00556ee55009",
          "url": "https://github.com/equinor/ert/commit/a95d0d2f2bd7bb6963d272c0f56635e0a47ed298"
        },
        "date": 1734590567766,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21899830226388506,
            "unit": "iter/sec",
            "range": "stddev: 0.03780022140777206",
            "extra": "mean: 4.566245444200002 sec\nrounds: 5"
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
          "id": "a509da2859f2c7feef650394ae67671b25aa7f92",
          "message": "Use f-strings for formatting",
          "timestamp": "2024-12-19T07:55:34+01:00",
          "tree_id": "09a6e64a25dc1118d6324099f0c6aebf9c41422b",
          "url": "https://github.com/equinor/ert/commit/a509da2859f2c7feef650394ae67671b25aa7f92"
        },
        "date": 1734591442472,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22300412341274142,
            "unit": "iter/sec",
            "range": "stddev: 0.01975973325220808",
            "extra": "mean: 4.484222016600006 sec\nrounds: 5"
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
          "id": "2b41fe21b9f34819bac83824140beeed7c9f6ec8",
          "message": "Type filter_env_dict and rename internals",
          "timestamp": "2024-12-19T08:01:03+01:00",
          "tree_id": "eb19d3614066231f2296ca5de0fd91012dd85948",
          "url": "https://github.com/equinor/ert/commit/2b41fe21b9f34819bac83824140beeed7c9f6ec8"
        },
        "date": 1734591772384,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2194161609945277,
            "unit": "iter/sec",
            "range": "stddev: 0.015882687725442296",
            "extra": "mean: 4.557549432400014 sec\nrounds: 5"
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
          "id": "e5c79532c79e4d58baf1884fb271a42f1eba68e9",
          "message": "Remove printing of snapshots in gui test",
          "timestamp": "2024-12-19T08:29:01+01:00",
          "tree_id": "6b7943ef213830c5bcd6106011c6ab9e37f5555b",
          "url": "https://github.com/equinor/ert/commit/e5c79532c79e4d58baf1884fb271a42f1eba68e9"
        },
        "date": 1734593448628,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21571393792067625,
            "unit": "iter/sec",
            "range": "stddev: 0.02173625803016719",
            "extra": "mean: 4.635769063599992 sec\nrounds: 5"
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
          "id": "a159fa7f0ca17724652f94786fff77346fa13c87",
          "message": "Move duplicate patching in test_everserver.py to mock_server fixture",
          "timestamp": "2024-12-19T10:57:39+02:00",
          "tree_id": "e8ba160d85ca2bfdaad309fda5f0bc8d82c145a7",
          "url": "https://github.com/equinor/ert/commit/a159fa7f0ca17724652f94786fff77346fa13c87"
        },
        "date": 1734598766393,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21820232936264464,
            "unit": "iter/sec",
            "range": "stddev: 0.05581156998013488",
            "extra": "mean: 4.582902496600002 sec\nrounds: 5"
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
          "id": "2a4b6bef24f8e7f8a3a7f02a5eb3972c00026696",
          "message": "Pretty print manifest.json\n\njobs.json was already effectively pretty-printed as it is overwritten at\na later stage, but now both writes of jobs.json are consistent in\nprinting pretty.",
          "timestamp": "2024-12-19T11:04:26+01:00",
          "tree_id": "2c8e9c362bcd60e04829f85c1a0415fd2b72bc62",
          "url": "https://github.com/equinor/ert/commit/2a4b6bef24f8e7f8a3a7f02a5eb3972c00026696"
        },
        "date": 1734602781828,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21900688318875033,
            "unit": "iter/sec",
            "range": "stddev: 0.04506683164528437",
            "extra": "mean: 4.566066533800006 sec\nrounds: 5"
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
          "id": "d9088eb8e2f26e5b38cc45160fdca49786b1a359",
          "message": "Validate triangular dist parameters on startup",
          "timestamp": "2024-12-19T14:17:49+01:00",
          "tree_id": "b3526ff487600adff34e2a0faf8628a631ebe8cd",
          "url": "https://github.com/equinor/ert/commit/d9088eb8e2f26e5b38cc45160fdca49786b1a359"
        },
        "date": 1734614379815,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2149859993083223,
            "unit": "iter/sec",
            "range": "stddev: 0.022294367364244606",
            "extra": "mean: 4.651465691800001 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9ba5a2829e4cbd87e9e8085727e206934375a99a",
          "message": "Change obs_and_responses test to benchmark",
          "timestamp": "2024-12-19T14:12:33Z",
          "tree_id": "d525a5930d19157d43e3d46e6c591b23c849548e",
          "url": "https://github.com/equinor/ert/commit/9ba5a2829e4cbd87e9e8085727e206934375a99a"
        },
        "date": 1734617692780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2147107684460768,
            "unit": "iter/sec",
            "range": "stddev: 0.12342531081203442",
            "extra": "mean: 4.6574282568 sec\nrounds: 5"
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
          "id": "e64449681453b0945c0673732450f1c3c8a8155a",
          "message": "Simplify tests and remove uncessary config files everest/mathfunc (#9434)\n\nCleanup Everest config files and tests in mathfunc",
          "timestamp": "2024-12-19T18:03:08+01:00",
          "tree_id": "c9c3799198de222b500f0e28c050f99b5aa669f9",
          "url": "https://github.com/equinor/ert/commit/e64449681453b0945c0673732450f1c3c8a8155a"
        },
        "date": 1734627898409,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21718485304406093,
            "unit": "iter/sec",
            "range": "stddev: 0.04668860467064045",
            "extra": "mean: 4.6043726622000065 sec\nrounds: 5"
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
          "id": "2e9c71a22e8fa11a7d6aa5d7ed5e5deeeed0862f",
          "message": "Add line numbers when validating everest config files",
          "timestamp": "2024-12-20T07:40:03+01:00",
          "tree_id": "6494a356704c566840a21f42cbdb516d95475e29",
          "url": "https://github.com/equinor/ert/commit/2e9c71a22e8fa11a7d6aa5d7ed5e5deeeed0862f"
        },
        "date": 1734676909281,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2201797334773787,
            "unit": "iter/sec",
            "range": "stddev: 0.02277579060793452",
            "extra": "mean: 4.5417440752000005 sec\nrounds: 5"
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
          "id": "1889e2ab1cb9bc44d4791985d181181a253caf4e",
          "message": "Merge EGG.DATA and EGG_FLOW.DATA\n\nRemoving QUIESC from EGG.DATA makes it possible to merge these two decks into one,\nand fixes an otherwise non-functional EGG_FLOW setup. This allows future CI tests on\nEverest with flow.",
          "timestamp": "2024-12-20T12:05:22+01:00",
          "tree_id": "ec7e28c8c9ca965f16effa8c9d42c5a51c83a2b4",
          "url": "https://github.com/equinor/ert/commit/1889e2ab1cb9bc44d4791985d181181a253caf4e"
        },
        "date": 1734692842806,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21489655153830137,
            "unit": "iter/sec",
            "range": "stddev: 0.04711308683411575",
            "extra": "mean: 4.653401801199999 sec\nrounds: 5"
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
          "id": "1bd4fe1294258050dd495e9a33ba3d23f47149b7",
          "message": "Remove unused async_run",
          "timestamp": "2024-12-20T14:06:05+01:00",
          "tree_id": "85bf2c979daf0ca04f6fd2a58006b95445cbe925",
          "url": "https://github.com/equinor/ert/commit/1bd4fe1294258050dd495e9a33ba3d23f47149b7"
        },
        "date": 1734700072391,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21569730252161742,
            "unit": "iter/sec",
            "range": "stddev: 0.03573023895218398",
            "extra": "mean: 4.636126591800002 sec\nrounds: 5"
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
          "id": "8fa3040800cc5253dc1ed5a34c8ceb32281373c7",
          "message": "Remove outdated logging handle",
          "timestamp": "2024-12-20T14:07:24+01:00",
          "tree_id": "0d92b4c9a2b2f21d743ac9e00697aa34bc0a79df",
          "url": "https://github.com/equinor/ert/commit/8fa3040800cc5253dc1ed5a34c8ceb32281373c7"
        },
        "date": 1734700155622,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21838175275939375,
            "unit": "iter/sec",
            "range": "stddev: 0.03988909457111767",
            "extra": "mean: 4.579137164000002 sec\nrounds: 5"
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
          "id": "722e0fc805cde27add2a3ad3344fcfc99fdf5b10",
          "message": "EverestRunModel: sanitize realization var names",
          "timestamp": "2024-12-20T14:27:56+01:00",
          "tree_id": "ed41627d26d55693d518eb54f9bad50b37092805",
          "url": "https://github.com/equinor/ert/commit/722e0fc805cde27add2a3ad3344fcfc99fdf5b10"
        },
        "date": 1734701386950,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21948512170606432,
            "unit": "iter/sec",
            "range": "stddev: 0.019461140551392343",
            "extra": "mean: 4.5561174817999985 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jparu@equinor.com",
            "name": "xjules",
            "username": "xjules"
          },
          "committer": {
            "email": "jparu@equinor.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "distinct": true,
          "id": "9b1a01e384a8ee51ba79179633afa44f85f9d070",
          "message": "Implementing router-dealer pattern with custom acknowledgments with zmq\n\n - dealers always wait for acknowledgment from the evaluator\n - removing websockets and wait_for_evaluator\n - Settup encryption with curve\n - each dealer (client, dispatcher) will get a unique name\n - Monitor is an advanced version Client\n - _server_started.wait() is to signal that zmq router socket is bound\n - Use TCP protocol only when using LSF, SLURM or TORQUE queues\n -- Use ipc_protocol when using LOCAL driver\n - Remove certificate\n - Remove synced _send from Client\n - Remove cert generator\n - Remove ClientConnectionClosedOK\n - Add test for new connection while closing down evaluator\n - Add test for handle dispatcher and dispatcher messages in evaluator\n - Add tests for ipc and tcp ee config\n - Add test for clear connect and disconnect of Monitor\n - Set a a correct protocol for everestserver",
          "timestamp": "2024-12-20T15:10:48+01:00",
          "tree_id": "97f8232e6ba8bbfc1633b74ddd045360372e50f5",
          "url": "https://github.com/equinor/ert/commit/9b1a01e384a8ee51ba79179633afa44f85f9d070"
        },
        "date": 1734703955491,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2213877592052432,
            "unit": "iter/sec",
            "range": "stddev: 0.03837322182960567",
            "extra": "mean: 4.516961568200003 sec\nrounds: 5"
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
          "id": "b13ad454dbeceaee6a3f3470c9d95693d0ca9a02",
          "message": "Remove deprecated torque options\n\nThis commit removes the deprecated torque/openpbs queue options:\n* QUEUE_QUERY_TIMEOUT\n* NUM_NODES\n* NUM_CPUS_PER_NODE\n* QSTAT_OPTIONS\n* MEMORY_PER_JOB",
          "timestamp": "2024-12-20T15:29:08+01:00",
          "tree_id": "8bb6563900111aa9740d7bf5d3083038ff475345",
          "url": "https://github.com/equinor/ert/commit/b13ad454dbeceaee6a3f3470c9d95693d0ca9a02"
        },
        "date": 1734705055699,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21959705996330614,
            "unit": "iter/sec",
            "range": "stddev: 0.023333546213795374",
            "extra": "mean: 4.553795028800005 sec\nrounds: 5"
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
          "id": "855c517890b2342f82907a11ae519aec2b50d885",
          "message": "Remove unused type ignore.",
          "timestamp": "2024-12-23T18:03:09+02:00",
          "tree_id": "78a1ee71b253db603121844eea9e740b147e1662",
          "url": "https://github.com/equinor/ert/commit/855c517890b2342f82907a11ae519aec2b50d885"
        },
        "date": 1734969896128,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21858073960126662,
            "unit": "iter/sec",
            "range": "stddev: 0.008085150111211182",
            "extra": "mean: 4.574968507400024 sec\nrounds: 5"
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
          "id": "9ba6c47784c8c7ad59a84ea6a645f42c759721eb",
          "message": "Avoid warning from pytest\n\nPytestWarning: Value of environment variable ERT_STORAGE_ENS_PATH type\nshould be str, but got PosixPath('/.../storage') (type: PosixPath);\nconverted to str implicitly",
          "timestamp": "2025-01-02T09:05:09+01:00",
          "tree_id": "f12f6a458868d0303e7fc7d5524ade45dd29d3b8",
          "url": "https://github.com/equinor/ert/commit/9ba6c47784c8c7ad59a84ea6a645f42c759721eb"
        },
        "date": 1735805219779,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22087436811815722,
            "unit": "iter/sec",
            "range": "stddev: 0.015636184030795086",
            "extra": "mean: 4.527460603599996 sec\nrounds: 5"
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
          "id": "929d59b73cea011aafc95bd62a950d963928cf64",
          "message": "Avoid using deprecated --target-case in tests",
          "timestamp": "2025-01-02T09:05:27+01:00",
          "tree_id": "b6dc1b1cae0e7a3c6f0c6767c52da7b975f55dd5",
          "url": "https://github.com/equinor/ert/commit/929d59b73cea011aafc95bd62a950d963928cf64"
        },
        "date": 1735805240505,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2130313133209388,
            "unit": "iter/sec",
            "range": "stddev: 0.01825512749633529",
            "extra": "mean: 4.694145590200003 sec\nrounds: 5"
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
          "id": "45a4b7ca72b6bdda0a777d7a4712d19d97ab32b6",
          "message": "Workaround Python bug for shutil.copytree() exceptions\n\nSee https://github.com/python/cpython/issues/102931\n\nThis PR will detect if the bug is triggered, and massage the data\naccordingly. Patch is prepared upstream destined for Python 3.14.\n\nExisting tests are split for readability and preciseness, and extended\nto test that the workaround is performing as it should.",
          "timestamp": "2025-01-02T09:50:23+01:00",
          "tree_id": "c547f6e90012d98758f3c8a6061b15f77febb069",
          "url": "https://github.com/equinor/ert/commit/45a4b7ca72b6bdda0a777d7a4712d19d97ab32b6"
        },
        "date": 1735807930663,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2204107451657714,
            "unit": "iter/sec",
            "range": "stddev: 0.0216063470528396",
            "extra": "mean: 4.536983889999999 sec\nrounds: 5"
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
          "id": "9e2e32f88af2a4eec85cacdaddc47ba4af2de227",
          "message": "Avoid deprecated utcnow()",
          "timestamp": "2025-01-02T12:33:28+01:00",
          "tree_id": "46ad2b0faf1c940fa80a69961197d31097d0cbb9",
          "url": "https://github.com/equinor/ert/commit/9e2e32f88af2a4eec85cacdaddc47ba4af2de227"
        },
        "date": 1735817713031,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21773301885494384,
            "unit": "iter/sec",
            "range": "stddev: 0.03724143083640846",
            "extra": "mean: 4.592780669000007 sec\nrounds: 5"
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
          "id": "135c3c9053c91364bc957d979c42a8365c124c6a",
          "message": "Avoid repeated logging for each cluster in misfit analysis",
          "timestamp": "2025-01-02T15:48:01+01:00",
          "tree_id": "f1ed2b06ce0701e3b0e12ccf96829f5a95274d16",
          "url": "https://github.com/equinor/ert/commit/135c3c9053c91364bc957d979c42a8365c124c6a"
        },
        "date": 1735829396671,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21640452699990395,
            "unit": "iter/sec",
            "range": "stddev: 0.018154236135735673",
            "extra": "mean: 4.620975419799993 sec\nrounds: 5"
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
          "id": "7e8d20cdc4001980021b6b51b87e76e53a5c6dfc",
          "message": "Change needed for updated everest-models",
          "timestamp": "2025-01-03T09:23:55+01:00",
          "tree_id": "68e380011fb2204e36e75a0dd9ff56f48f40d243",
          "url": "https://github.com/equinor/ert/commit/7e8d20cdc4001980021b6b51b87e76e53a5c6dfc"
        },
        "date": 1735892743897,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22004640687181604,
            "unit": "iter/sec",
            "range": "stddev: 0.023821548294085563",
            "extra": "mean: 4.5444959280000035 sec\nrounds: 5"
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
          "id": "a372383de1d69a2b0da463f1fd6897e5cd50b3e0",
          "message": "Pin iterative_ensemble_smoother\n\nTests are not working with 0.3.0",
          "timestamp": "2025-01-03T13:11:29+01:00",
          "tree_id": "8a9d3cc3d4034939aa0154276fc62cf534a14b81",
          "url": "https://github.com/equinor/ert/commit/a372383de1d69a2b0da463f1fd6897e5cd50b3e0"
        },
        "date": 1735906397422,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21740204916053793,
            "unit": "iter/sec",
            "range": "stddev: 0.03137568113211991",
            "extra": "mean: 4.599772651 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}