window.BENCHMARK_DATA = {
  "lastUpdate": 1732196520379,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "a1a13cc1eb0aa3b8ac77bdc995f414c5e3428c08",
          "message": "Simplify test logic",
          "timestamp": "2024-11-15T21:42:46+09:00",
          "tree_id": "419e2444034fb044dc4d8dc1dca0706fee4905aa",
          "url": "https://github.com/equinor/ert/commit/a1a13cc1eb0aa3b8ac77bdc995f414c5e3428c08"
        },
        "date": 1731674679407,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19076791500288928,
            "unit": "iter/sec",
            "range": "stddev: 0.02509802690880647",
            "extra": "mean: 5.241971638599995 sec\nrounds: 5"
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
          "id": "58c1b1fcc5b0303a4f8e3d424f75d9a745264632",
          "message": "Unpin websockets",
          "timestamp": "2024-11-15T13:49:05+01:00",
          "tree_id": "1a4cced8ba301b67db19805ff4f42ab257593426",
          "url": "https://github.com/equinor/ert/commit/58c1b1fcc5b0303a4f8e3d424f75d9a745264632"
        },
        "date": 1731675052955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1944160101319624,
            "unit": "iter/sec",
            "range": "stddev: 0.03315314304280298",
            "extra": "mean: 5.143609311400008 sec\nrounds: 5"
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
          "id": "90c7a4cd29b5548a83d0ce81b3952240e7dddeff",
          "message": "Fix walltime computation bug\n\nBecause we logged finished step events before the current batch\nof events were processed, there were situations where the snapshot from a previous\nfailed realization was out of date.\n\nThis solves that problem. There is still a residual problem that will surface if\nwalltime is to be computed for all steps: If a failure event for a step and its\nsuccess event from the subsequent run is in the same batch of events, we can get\na negative walltime computed. This problem is ignored as the walltime computation\nis only interesting (currently) for walltimes > 120 seconds, where this cannot\nhappen (max time between batches is 2 seconds).",
          "timestamp": "2024-11-15T14:01:27+01:00",
          "tree_id": "a69defab2b5866ba5c7e92ff1c16d532625a93b8",
          "url": "https://github.com/equinor/ert/commit/90c7a4cd29b5548a83d0ce81b3952240e7dddeff"
        },
        "date": 1731675806223,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1938057308711761,
            "unit": "iter/sec",
            "range": "stddev: 0.0323433171968204",
            "extra": "mean: 5.159806139400007 sec\nrounds: 5"
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
          "id": "cc33a48341d019c88b1b5e414e99b85317f762ca",
          "message": "Remove comment",
          "timestamp": "2024-11-15T14:20:00+01:00",
          "tree_id": "f66f96bb3ee5ba8e89d7df436915baa2c09cc804",
          "url": "https://github.com/equinor/ert/commit/cc33a48341d019c88b1b5e414e99b85317f762ca"
        },
        "date": 1731676918720,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1910787636976867,
            "unit": "iter/sec",
            "range": "stddev: 0.02442527226961437",
            "extra": "mean: 5.233443950799995 sec\nrounds: 5"
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
          "id": "c46b06127ac5a449e221096e0771b50481676a92",
          "message": "Leave test machine usable when running rapid-tests\n\nRunning with a lower CPU priority helps this",
          "timestamp": "2024-11-16T15:54:48+01:00",
          "tree_id": "4c0f8bf80146edf417e9ec9b55732972ca847f6b",
          "url": "https://github.com/equinor/ert/commit/c46b06127ac5a449e221096e0771b50481676a92"
        },
        "date": 1731768999463,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19496214685337976,
            "unit": "iter/sec",
            "range": "stddev: 0.020167511008323178",
            "extra": "mean: 5.129200802000014 sec\nrounds: 5"
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
          "id": "d58862b596cd6fc58039f3eeff3b43d240e4eeb8",
          "message": "Remove outdated git lfs instruction\n\n\"rpm -qa | grep git-lfs\" reveals this is in place",
          "timestamp": "2024-11-18T07:28:07+01:00",
          "tree_id": "8e22e8306b42ae84860c58112f5a1e15a74c8e05",
          "url": "https://github.com/equinor/ert/commit/d58862b596cd6fc58039f3eeff3b43d240e4eeb8"
        },
        "date": 1731911398264,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19320606802078388,
            "unit": "iter/sec",
            "range": "stddev: 0.02445986601585514",
            "extra": "mean: 5.1758208747999905 sec\nrounds: 5"
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
          "id": "dbb6321c6df23be68d3e486a4d13a6e95a8928e1",
          "message": "Add test for mean calculation in dark_storage common",
          "timestamp": "2024-11-18T08:02:15+01:00",
          "tree_id": "37f1993492c76236526953de60cb4d0600c60e48",
          "url": "https://github.com/equinor/ert/commit/dbb6321c6df23be68d3e486a4d13a6e95a8928e1"
        },
        "date": 1731913442729,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19064655449599555,
            "unit": "iter/sec",
            "range": "stddev: 0.02051725654738917",
            "extra": "mean: 5.245308537800009 sec\nrounds: 5"
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
          "id": "c4a2062e5a094b52f5c7c1ffbac132d6ecc1c6d3",
          "message": "Refactor usage of EverestConfig in everest server functionality",
          "timestamp": "2024-11-18T19:14:16+09:00",
          "tree_id": "2c8857c1297c0b21eff6c9ab62420b378b1ab698",
          "url": "https://github.com/equinor/ert/commit/c4a2062e5a094b52f5c7c1ffbac132d6ecc1c6d3"
        },
        "date": 1731924969313,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19563452425450104,
            "unit": "iter/sec",
            "range": "stddev: 0.04089270139270816",
            "extra": "mean: 5.111572222799998 sec\nrounds: 5"
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
          "id": "a02a4867455e2eef226483ad0e45188735e18d62",
          "message": "Remove ErtConfig and LibresFacade",
          "timestamp": "2024-11-18T11:57:17+01:00",
          "tree_id": "998cf8d3ae389628fba14b1aef664dfe8f944198",
          "url": "https://github.com/equinor/ert/commit/a02a4867455e2eef226483ad0e45188735e18d62"
        },
        "date": 1731927551755,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19135878581242316,
            "unit": "iter/sec",
            "range": "stddev: 0.03613657517165334",
            "extra": "mean: 5.225785666200016 sec\nrounds: 5"
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
          "id": "e2924f1b4f98ba583bcdd3f4afaa07fd128e1fe4",
          "message": "Remove flaky memory profiling test\n\nThis test is too hard to make robust on loaded hardware.",
          "timestamp": "2024-11-18T14:58:11+01:00",
          "tree_id": "030728307887e051648f9f7482cc4aa2bebae111",
          "url": "https://github.com/equinor/ert/commit/e2924f1b4f98ba583bcdd3f4afaa07fd128e1fe4"
        },
        "date": 1731938404760,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19352927575116952,
            "unit": "iter/sec",
            "range": "stddev: 0.03164605887585596",
            "extra": "mean: 5.167176883800005 sec\nrounds: 5"
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
          "id": "2475f7fcae76599512b0efc2f74f1edca184b316",
          "message": "Log which hostname had an OOM-situation",
          "timestamp": "2024-11-18T15:24:29+01:00",
          "tree_id": "9055cff49d09320367543cf2926c542dc2b0c4fc",
          "url": "https://github.com/equinor/ert/commit/2475f7fcae76599512b0efc2f74f1edca184b316"
        },
        "date": 1731939985122,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19130676982102993,
            "unit": "iter/sec",
            "range": "stddev: 0.05889932274208237",
            "extra": "mean: 5.227206548599997 sec\nrounds: 5"
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
          "id": "49dde01efb96dc9b19fef2c07b6417fac5dce60e",
          "message": "Fix incorrect batch ensemble getter",
          "timestamp": "2024-11-19T09:48:35+01:00",
          "tree_id": "de6901c4332e6c9fe553c6854838e05a661d0cbe",
          "url": "https://github.com/equinor/ert/commit/49dde01efb96dc9b19fef2c07b6417fac5dce60e"
        },
        "date": 1732006232480,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19461294387104142,
            "unit": "iter/sec",
            "range": "stddev: 0.02770199828680481",
            "extra": "mean: 5.138404363599994 sec\nrounds: 5"
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
          "id": "f38fd9d30f3e91e0cc9b38d247ab17ca84e1dd68",
          "message": "Perform validation earlier by returning EnOptConfig instead of dict  (#9249)\n\nPerform earlier validation on the EnOptConfig",
          "timestamp": "2024-11-19T10:01:30+01:00",
          "tree_id": "5a0df2e7ea226799299a7c87f624ddb2e4e9d4c3",
          "url": "https://github.com/equinor/ert/commit/f38fd9d30f3e91e0cc9b38d247ab17ca84e1dd68"
        },
        "date": 1732007004879,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19363517092743143,
            "unit": "iter/sec",
            "range": "stddev: 0.038216885462063914",
            "extra": "mean: 5.164351058800003 sec\nrounds: 5"
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
          "id": "0d2c4c8c70f24ddbd293f60b0d97919e7bfcd47e",
          "message": "Instantiate ert config before starting everest server",
          "timestamp": "2024-11-19T19:25:11+09:00",
          "tree_id": "ed2b71c90e3b4b6337cd78205dcafc3cd84e29b4",
          "url": "https://github.com/equinor/ert/commit/0d2c4c8c70f24ddbd293f60b0d97919e7bfcd47e"
        },
        "date": 1732012021877,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1925922317597761,
            "unit": "iter/sec",
            "range": "stddev: 0.03895071287625285",
            "extra": "mean: 5.192317420400002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "86219529+AugustoMagalhaes@users.noreply.github.com",
            "name": "AugustoMagalhaes",
            "username": "AugustoMagalhaes"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2a40bed18eca037e87919ac8f021609dfa441a0b",
          "message": "Highlights selected tab in main window\n\nCo-authored-by: Eivind Jahren <ejah@equinor.com>",
          "timestamp": "2024-11-19T13:45:02+01:00",
          "tree_id": "d5e58e7af829ed09785811aeb41882a413c05ffa",
          "url": "https://github.com/equinor/ert/commit/2a40bed18eca037e87919ac8f021609dfa441a0b"
        },
        "date": 1732020425849,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.192367426625743,
            "unit": "iter/sec",
            "range": "stddev: 0.03408702116653935",
            "extra": "mean: 5.1983852856 sec\nrounds: 5"
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
          "id": "d07a7e7869ccf998597e36d46766e009ac31b512",
          "message": "Update everest install command in ReadMe (#9255)\n\nUpdate everest install command in ReadMe docs",
          "timestamp": "2024-11-19T15:41:09+01:00",
          "tree_id": "41feea775dc97c00633b026c48dac62591993aea",
          "url": "https://github.com/equinor/ert/commit/d07a7e7869ccf998597e36d46766e009ac31b512"
        },
        "date": 1732027380280,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18882093614298373,
            "unit": "iter/sec",
            "range": "stddev: 0.03430320971169947",
            "extra": "mean: 5.296022890400008 sec\nrounds: 5"
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
          "id": "3b603d448b4d996f9a6414580688bd0209817802",
          "message": "Add api snapshot test w/ summary data",
          "timestamp": "2024-11-20T08:02:13+01:00",
          "tree_id": "31bb5d47e341d0d07a704b0636b6904b9e1d2c9f",
          "url": "https://github.com/equinor/ert/commit/3b603d448b4d996f9a6414580688bd0209817802"
        },
        "date": 1732086241256,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19876168725926224,
            "unit": "iter/sec",
            "range": "stddev: 0.018711861136174072",
            "extra": "mean: 5.0311506899999925 sec\nrounds: 5"
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
          "id": "b555902c5d7de1dc9cb426d5350a9958c454fbe2",
          "message": "get_ensemble_responses without get_ensemble_state",
          "timestamp": "2024-11-20T11:33:43+01:00",
          "tree_id": "b4782fa4b3c102abc97c5f4c27e298ed5082dcc6",
          "url": "https://github.com/equinor/ert/commit/b555902c5d7de1dc9cb426d5350a9958c454fbe2"
        },
        "date": 1732098951450,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19221839416979355,
            "unit": "iter/sec",
            "range": "stddev: 0.0150592879566697",
            "extra": "mean: 5.202415743399996 sec\nrounds: 5"
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
          "id": "a8f4d7312692eda17a8f3634aad5dc82087207dd",
          "message": "Fix ecl not able to parce errors from MPI runs (#9248)",
          "timestamp": "2024-11-20T14:40:06+01:00",
          "tree_id": "26a66b3d6165716475936b9b19760bf4fd8a988d",
          "url": "https://github.com/equinor/ert/commit/a8f4d7312692eda17a8f3634aad5dc82087207dd"
        },
        "date": 1732110119771,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1939413864438311,
            "unit": "iter/sec",
            "range": "stddev: 0.021801768775517923",
            "extra": "mean: 5.156197026000006 sec\nrounds: 5"
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
          "id": "57e6b70db861f019241a999010b09279469527f9",
          "message": "Fix issue where dark_storage did not handle empty responses",
          "timestamp": "2024-11-20T14:41:37+01:00",
          "tree_id": "8f1ebbb6f8a23dc9fd77cfaa951bd0015fb14af7",
          "url": "https://github.com/equinor/ert/commit/57e6b70db861f019241a999010b09279469527f9"
        },
        "date": 1732110209947,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18915693655015092,
            "unit": "iter/sec",
            "range": "stddev: 0.01876167033183068",
            "extra": "mean: 5.286615538599989 sec\nrounds: 5"
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
          "id": "12144e3fd0022424b4228cf5599e672ed0d469e3",
          "message": "Remove loading the same workflow twice (#9257)",
          "timestamp": "2024-11-21T08:11:02+01:00",
          "tree_id": "71c8f251212dba9c5653171d585661b60de4bcd4",
          "url": "https://github.com/equinor/ert/commit/12144e3fd0022424b4228cf5599e672ed0d469e3"
        },
        "date": 1732173174378,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1948615488916678,
            "unit": "iter/sec",
            "range": "stddev: 0.018399556017963052",
            "extra": "mean: 5.1318487699999995 sec\nrounds: 5"
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
          "id": "19c7192081cfe0f44ffc9e22ad2a24afc3a3eb7e",
          "message": "Degrade logged warning to info\n\nThis was a workaround to propagate this particular log through\na filter that only propagated warning. Now info logs are also\npropagated so this workaround is no longer needed.",
          "timestamp": "2024-11-21T08:17:23+01:00",
          "tree_id": "be05b339489c834c5e849b27a1f644714777aa49",
          "url": "https://github.com/equinor/ert/commit/19c7192081cfe0f44ffc9e22ad2a24afc3a3eb7e"
        },
        "date": 1732173556981,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19129578869824207,
            "unit": "iter/sec",
            "range": "stddev: 0.022726353756363466",
            "extra": "mean: 5.227506610599994 sec\nrounds: 5"
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
          "id": "81e21d9918847646016afd255a47c8405705a776",
          "message": "Aggregate logging of sampling information\n\nA separate log entry for each realization does not provide much\nextra value",
          "timestamp": "2024-11-21T08:18:12+01:00",
          "tree_id": "2d0e93b6169c25bffd3ecbe953512888fb92defa",
          "url": "https://github.com/equinor/ert/commit/81e21d9918847646016afd255a47c8405705a776"
        },
        "date": 1732173600891,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1934878001944904,
            "unit": "iter/sec",
            "range": "stddev: 0.03492326630468336",
            "extra": "mean: 5.168284506800006 sec\nrounds: 5"
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
          "id": "8b4d62ea5edbaa10b33a331e898c47cc2e3e7e3e",
          "message": "Pin pydantic version",
          "timestamp": "2024-11-21T09:05:45+01:00",
          "tree_id": "cf5a2b7d23d84dc82bb4daff2db2ef6b65bfe0cb",
          "url": "https://github.com/equinor/ert/commit/8b4d62ea5edbaa10b33a331e898c47cc2e3e7e3e"
        },
        "date": 1732176457188,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1869556093408083,
            "unit": "iter/sec",
            "range": "stddev: 0.06031120445795523",
            "extra": "mean: 5.348863313199996 sec\nrounds: 5"
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
          "id": "46f7ba390923c646ae33905db5a77fbb21761b07",
          "message": "Remove unused test-data",
          "timestamp": "2024-11-21T10:26:28+01:00",
          "tree_id": "938237efb75e70bbc0eb1daa76c7b378db407d37",
          "url": "https://github.com/equinor/ert/commit/46f7ba390923c646ae33905db5a77fbb21761b07"
        },
        "date": 1732181300594,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19101803610871523,
            "unit": "iter/sec",
            "range": "stddev: 0.04454005559099521",
            "extra": "mean: 5.235107743600002 sec\nrounds: 5"
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
          "id": "33cc2c2c6ccfce36bda3e3fc9f35acebe771b7b0",
          "message": "Test field with FORWARD_INIT:False",
          "timestamp": "2024-11-21T10:30:12+01:00",
          "tree_id": "d2af8bbeb81f635379f68d46e0851ac9fcffe60f",
          "url": "https://github.com/equinor/ert/commit/33cc2c2c6ccfce36bda3e3fc9f35acebe771b7b0"
        },
        "date": 1732181523045,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19352657535572754,
            "unit": "iter/sec",
            "range": "stddev: 0.024763611003450363",
            "extra": "mean: 5.167248984600008 sec\nrounds: 5"
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
          "id": "b5f9caa79234ef6f17a2208f83200babb25dcb46",
          "message": "Put back flaky tag for test_logging_setup",
          "timestamp": "2024-11-21T18:35:39+09:00",
          "tree_id": "7123662106b62d428fafdc32b7c77dc6c4a629d2",
          "url": "https://github.com/equinor/ert/commit/b5f9caa79234ef6f17a2208f83200babb25dcb46"
        },
        "date": 1732181851242,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1906726475798551,
            "unit": "iter/sec",
            "range": "stddev: 0.021604003101121476",
            "extra": "mean: 5.244590730200002 sec\nrounds: 5"
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
          "id": "e7640838cac5784f9d40fefcd3d8caa74c9b5548",
          "message": "Avoid forward model crash after MPI NOSIM E100 run\n\nThis fixes a regression from 2453a2c.",
          "timestamp": "2024-11-21T11:39:04+01:00",
          "tree_id": "b9cea9f34e4d59053df4c511b7413986474c8d8c",
          "url": "https://github.com/equinor/ert/commit/e7640838cac5784f9d40fefcd3d8caa74c9b5548"
        },
        "date": 1732185666635,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1907670810820264,
            "unit": "iter/sec",
            "range": "stddev: 0.036638804131725995",
            "extra": "mean: 5.241994553400008 sec\nrounds: 5"
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
            "email": "yngve-sk@users.noreply.github.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "distinct": true,
          "id": "77cf86fc88bf9178bee2e7b2bcefa6178934d34d",
          "message": "Increase number of examples for stateful storage test",
          "timestamp": "2024-11-21T13:49:34+01:00",
          "tree_id": "34343a1e7384a8ee56f6feb0e37d047a405fd7a5",
          "url": "https://github.com/equinor/ert/commit/77cf86fc88bf9178bee2e7b2bcefa6178934d34d"
        },
        "date": 1732193486007,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19013579343634682,
            "unit": "iter/sec",
            "range": "stddev: 0.05824185461817928",
            "extra": "mean: 5.259398990200009 sec\nrounds: 5"
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
          "id": "01063e214ad939b6c5ea55dddf4bc332ac750d07",
          "message": "Remove unused dependencies dask, deprecation, testpath",
          "timestamp": "2024-11-21T14:40:02+01:00",
          "tree_id": "b0c2fb96a44655a5dc29174c4936862f506f2870",
          "url": "https://github.com/equinor/ert/commit/01063e214ad939b6c5ea55dddf4bc332ac750d07"
        },
        "date": 1732196519460,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1980623859344135,
            "unit": "iter/sec",
            "range": "stddev: 0.024672064553451997",
            "extra": "mean: 5.048914236200005 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}