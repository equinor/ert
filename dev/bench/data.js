window.BENCHMARK_DATA = {
  "lastUpdate": 1732098951936,
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
          "id": "a1a028c3aacbd0ba1e813b4c9842bbaf108ae66d",
          "message": "Disregard expected_objectives from OptimalResults\n\nthe calculation within seba does not make sense",
          "timestamp": "2024-11-14T09:03:31+01:00",
          "tree_id": "434623a1d3e6a0ff07e83e6005c3dbfa8a35da2b",
          "url": "https://github.com/equinor/ert/commit/a1a028c3aacbd0ba1e813b4c9842bbaf108ae66d"
        },
        "date": 1731571528260,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19223385464088083,
            "unit": "iter/sec",
            "range": "stddev: 0.02040400479608584",
            "extra": "mean: 5.201997337399996 sec\nrounds: 5"
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
          "id": "1b235cf54a592a6b3be1b0c7ba0a95570115d288",
          "message": "Use sh for all code blocks in readme",
          "timestamp": "2024-11-14T10:26:21+01:00",
          "tree_id": "5a88f390f18b00da23698eb80ede1d040f3809c9",
          "url": "https://github.com/equinor/ert/commit/1b235cf54a592a6b3be1b0c7ba0a95570115d288"
        },
        "date": 1731576497317,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18999392828727604,
            "unit": "iter/sec",
            "range": "stddev: 0.01782326892716323",
            "extra": "mean: 5.263326091600004 sec\nrounds: 5"
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
          "id": "fd6fa0574a8edc085d50d2872d16b19646217dfd",
          "message": "Fix typing for batching interval",
          "timestamp": "2024-11-14T10:42:41+01:00",
          "tree_id": "6f55cfa43074a52a9439ac804c8563b1253dd420",
          "url": "https://github.com/equinor/ert/commit/fd6fa0574a8edc085d50d2872d16b19646217dfd"
        },
        "date": 1731577474307,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19463709917666028,
            "unit": "iter/sec",
            "range": "stddev: 0.010724499538523202",
            "extra": "mean: 5.137766665400005 sec\nrounds: 5"
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
          "id": "91310f99e95e135cc2e63fbce0640033352e9c7b",
          "message": "Run ruff",
          "timestamp": "2024-11-14T11:54:05+01:00",
          "tree_id": "b9864e9c010671e7d6344582cb8b114e47bf6034",
          "url": "https://github.com/equinor/ert/commit/91310f99e95e135cc2e63fbce0640033352e9c7b"
        },
        "date": 1731581777665,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18832109962684743,
            "unit": "iter/sec",
            "range": "stddev: 0.06783166517792047",
            "extra": "mean: 5.310079444000007 sec\nrounds: 5"
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
          "id": "9a3c864607c891791a023e54abe6cea4706bf9ae",
          "message": "Convert ErtConfig to dataclass",
          "timestamp": "2024-11-14T12:28:20+01:00",
          "tree_id": "9afa168d7d375b58f8ba9a20880ff18dafa4596c",
          "url": "https://github.com/equinor/ert/commit/9a3c864607c891791a023e54abe6cea4706bf9ae"
        },
        "date": 1731583824251,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1924551332956283,
            "unit": "iter/sec",
            "range": "stddev: 0.03686378717714496",
            "extra": "mean: 5.196016249999997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sted@equinor.com",
            "name": "StephanDeHoop",
            "username": "StephanDeHoop"
          },
          "committer": {
            "email": "stephan.dehoop@tno.nl",
            "name": "Stephan de Hoop",
            "username": "StephanDeHoop"
          },
          "distinct": true,
          "id": "137bdc64b0be0f9d300720e487458226dc4ee1e4",
          "message": "Use everest.strings instead of hardcoded",
          "timestamp": "2024-11-14T12:52:17+01:00",
          "tree_id": "6ef924718b0ff0a5b4da73391879d4072fb7d11d",
          "url": "https://github.com/equinor/ert/commit/137bdc64b0be0f9d300720e487458226dc4ee1e4"
        },
        "date": 1731585249027,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1954745562795878,
            "unit": "iter/sec",
            "range": "stddev: 0.021056245468429068",
            "extra": "mean: 5.115755313800008 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "augustommg@gmail.com",
            "name": "AugustoMagalhaes",
            "username": "AugustoMagalhaes"
          },
          "committer": {
            "email": "dan.sava42@gmail.com",
            "name": "Dan Sava",
            "username": "DanSava"
          },
          "distinct": true,
          "id": "120659c388e9419327c899d34f1c4a346b551037",
          "message": "Adds test for invalid install_data templates",
          "timestamp": "2024-11-14T21:11:05+09:00",
          "tree_id": "0c0948dcaf4f5dee5b224c55944eec961484d2cf",
          "url": "https://github.com/equinor/ert/commit/120659c388e9419327c899d34f1c4a346b551037"
        },
        "date": 1731586385230,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1919442611014824,
            "unit": "iter/sec",
            "range": "stddev: 0.029696380098208634",
            "extra": "mean: 5.209845786800014 sec\nrounds: 5"
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
          "id": "79aad486f76bbb27746c34de4c29758b028a93a4",
          "message": "Fix bug where Everest would not start with relative paths",
          "timestamp": "2024-11-14T13:22:04+01:00",
          "tree_id": "15e370783b3ae0b9b823688d0576f5f9acef1810",
          "url": "https://github.com/equinor/ert/commit/79aad486f76bbb27746c34de4c29758b028a93a4"
        },
        "date": 1731587032787,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19476826957680538,
            "unit": "iter/sec",
            "range": "stddev: 0.027146648481339533",
            "extra": "mean: 5.134306538600003 sec\nrounds: 5"
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
          "id": "9819a5b03255355abd69d16667bd95ae69e0afbc",
          "message": "Fix flaky rightclick plot-button test",
          "timestamp": "2024-11-14T13:40:12+01:00",
          "tree_id": "77d974138c33d242d929ff7cd5a11ae2e83b941e",
          "url": "https://github.com/equinor/ert/commit/9819a5b03255355abd69d16667bd95ae69e0afbc"
        },
        "date": 1731588134861,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19231872867327135,
            "unit": "iter/sec",
            "range": "stddev: 0.02460795550748442",
            "extra": "mean: 5.199701593800006 sec\nrounds: 5"
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
          "id": "e544ddcdc19898ef7aa6a8a198ebd4fdcf99e086",
          "message": "Fix faulty validation for maintained forward model objectives.",
          "timestamp": "2024-11-15T15:45:23+09:00",
          "tree_id": "7d5cff4b0ff8f6749f7ba2f42269159f2831d373",
          "url": "https://github.com/equinor/ert/commit/e544ddcdc19898ef7aa6a8a198ebd4fdcf99e086"
        },
        "date": 1731653254257,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18910558466570274,
            "unit": "iter/sec",
            "range": "stddev: 0.08942009358555235",
            "extra": "mean: 5.288051126400001 sec\nrounds: 5"
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
          "id": "ffaa27ab2532153f0971dab394bf825c1b56ad54",
          "message": "Extend save parameters to handle multiple realizations\n\nAdd test that uses the new functionality and also documents\r\nsome troublesome behavior of adaptive localization.",
          "timestamp": "2024-11-15T09:45:29+01:00",
          "tree_id": "a487246ff4b0bf535714a7434410a52ae809d1b3",
          "url": "https://github.com/equinor/ert/commit/ffaa27ab2532153f0971dab394bf825c1b56ad54"
        },
        "date": 1731660444481,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19541078065371262,
            "unit": "iter/sec",
            "range": "stddev: 0.008124962162825272",
            "extra": "mean: 5.117424927399986 sec\nrounds: 5"
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
          "id": "76125352e552ab45a2ff9c6c0a6f6c8745b7adf5",
          "message": "Fetch all tags in readthedocs workflow",
          "timestamp": "2024-11-15T11:48:13+01:00",
          "tree_id": "e33af86cc33b24b14b1ec322a10661ec8c7bead1",
          "url": "https://github.com/equinor/ert/commit/76125352e552ab45a2ff9c6c0a6f6c8745b7adf5"
        },
        "date": 1731667816815,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19012768757649334,
            "unit": "iter/sec",
            "range": "stddev: 0.041726213055851856",
            "extra": "mean: 5.2596232182 sec\nrounds: 5"
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
      }
    ]
  }
}