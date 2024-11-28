window.BENCHMARK_DATA = {
  "lastUpdate": 1732798696742,
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
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "4485f41bc30417fc3d4dd36bd5d43ce62099c373",
          "message": "Mark a test as flaky\n\nThis is a very rare occurence, reruns should be sufficient",
          "timestamp": "2024-11-25T15:24:58+01:00",
          "tree_id": "5a632a6a172c30eeae62f608cabe96b6c9226881",
          "url": "https://github.com/equinor/ert/commit/4485f41bc30417fc3d4dd36bd5d43ce62099c373"
        },
        "date": 1732544805756,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18920632357885153,
            "unit": "iter/sec",
            "range": "stddev: 0.11250853964733565",
            "extra": "mean: 5.2852356151999915 sec\nrounds: 5"
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
          "id": "6a4bcd2e86a23b9eaea0b8a1510694bdb5aab5b3",
          "message": "Ignore expected warnings in test",
          "timestamp": "2024-11-25T15:33:23+01:00",
          "tree_id": "bcf641507e38dd1b66694da593058f34a9ff391b",
          "url": "https://github.com/equinor/ert/commit/6a4bcd2e86a23b9eaea0b8a1510694bdb5aab5b3"
        },
        "date": 1732545313755,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19432645749479735,
            "unit": "iter/sec",
            "range": "stddev: 0.05835965963870035",
            "extra": "mean: 5.145979672000005 sec\nrounds: 5"
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
          "id": "c0cb78880bca1cdc045935f60f1e157ea60746b2",
          "message": "Remove report_steps\n\nThis was a remnant from when we could not load different summary\nfiles with different report_steps. This is no longer the case, so\nthis is just a complication.",
          "timestamp": "2024-11-25T16:46:14+01:00",
          "tree_id": "9ab87e9eff109b50a44da1ad82093c5c8b88667d",
          "url": "https://github.com/equinor/ert/commit/c0cb78880bca1cdc045935f60f1e157ea60746b2"
        },
        "date": 1732549685308,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19593875553165482,
            "unit": "iter/sec",
            "range": "stddev: 0.021031451449080443",
            "extra": "mean: 5.1036355584000095 sec\nrounds: 5"
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
          "id": "bb0318635e275ed426e597f8c91a63a581bd7565",
          "message": "Remove multi-realization save logic\n\n(testing)",
          "timestamp": "2024-11-26T08:46:16+01:00",
          "tree_id": "83ff05c81684940f2ddf2b1c85f0f065892318fd",
          "url": "https://github.com/equinor/ert/commit/bb0318635e275ed426e597f8c91a63a581bd7565"
        },
        "date": 1732607291835,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19237468187030143,
            "unit": "iter/sec",
            "range": "stddev: 0.05902346462649664",
            "extra": "mean: 5.198189232999994 sec\nrounds: 5"
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
          "id": "4b2c2f58090658fbc3ac85bebc4e48c48359e26d",
          "message": "Remove unused files from test-data/templating",
          "timestamp": "2024-11-26T10:22:33+01:00",
          "tree_id": "7d92cc687e995aa5bbd7a86187c8573f480d82de",
          "url": "https://github.com/equinor/ert/commit/4b2c2f58090658fbc3ac85bebc4e48c48359e26d"
        },
        "date": 1732613070919,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19472945167182465,
            "unit": "iter/sec",
            "range": "stddev: 0.05068456452254919",
            "extra": "mean: 5.135330025400004 sec\nrounds: 5"
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
          "id": "baccfe19b4be2ae90ca8da0ef477c9f0bf060456",
          "message": "Install everest deps when building readthedocs",
          "timestamp": "2024-11-26T12:15:48+01:00",
          "tree_id": "def7e8b8e07d63e37524ec49c0e5e1738fa0bb4b",
          "url": "https://github.com/equinor/ert/commit/baccfe19b4be2ae90ca8da0ef477c9f0bf060456"
        },
        "date": 1732619862701,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1951988419515447,
            "unit": "iter/sec",
            "range": "stddev: 0.029686839000548502",
            "extra": "mean: 5.122981212399997 sec\nrounds: 5"
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
          "id": "67e756dc9e56d73679877b190343600c21d397b5",
          "message": "Fix extra requirements for readthedocs build",
          "timestamp": "2024-11-26T12:40:01+01:00",
          "tree_id": "b3d67e2db5934891c684b168d6874e1f71cbb720",
          "url": "https://github.com/equinor/ert/commit/67e756dc9e56d73679877b190343600c21d397b5"
        },
        "date": 1732621313463,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1941991921705421,
            "unit": "iter/sec",
            "range": "stddev: 0.01528226801816358",
            "extra": "mean: 5.149352007200003 sec\nrounds: 5"
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
          "id": "71105ba7bc2b93473b87b1d36e2c7f8d52affbe2",
          "message": "Remove unused files from test-data/mocked_test_case",
          "timestamp": "2024-11-26T12:44:53+01:00",
          "tree_id": "9276b248a8a696337b683e2a3cce58906054a477",
          "url": "https://github.com/equinor/ert/commit/71105ba7bc2b93473b87b1d36e2c7f8d52affbe2"
        },
        "date": 1732621605497,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1915184988033237,
            "unit": "iter/sec",
            "range": "stddev: 0.035587246495269155",
            "extra": "mean: 5.22142772759999 sec\nrounds: 5"
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
          "id": "dc5ab03d3013b896c85ee095da20a89db375928b",
          "message": "Remove excess minimum realizations validation in baserunmodel\n\nThis commit removes the additional validation `validate_active_realizations_count` which checked that the number of  realizations to run was greater than the minimum required realizations count.\nThis validation is already done in `model_factory::_setup_ensemble_experiment`",
          "timestamp": "2024-11-26T14:12:41+01:00",
          "tree_id": "d248292d4d0450a5c669c1d0527461d91c22dcad",
          "url": "https://github.com/equinor/ert/commit/dc5ab03d3013b896c85ee095da20a89db375928b"
        },
        "date": 1732626874798,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19477501413344553,
            "unit": "iter/sec",
            "range": "stddev: 0.04043459587765664",
            "extra": "mean: 5.134128750800005 sec\nrounds: 5"
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
          "id": "d7829aca719d2016e4884ef38e464a318444f18b",
          "message": "Initialize ert config with plugins when starting webviz ert",
          "timestamp": "2024-11-26T14:30:14+01:00",
          "tree_id": "824133a85bc64e939f99759d96c5bb5ae006c2e7",
          "url": "https://github.com/equinor/ert/commit/d7829aca719d2016e4884ef38e464a318444f18b"
        },
        "date": 1732627926837,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1908070904512115,
            "unit": "iter/sec",
            "range": "stddev: 0.03657527773268285",
            "extra": "mean: 5.240895386199997 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "ca1806e59d1a6a45a18cbc4a3c4ca51af6b4db0b",
          "message": "Make bsub tests faster (#9350)\n\n* Remove needless wait in driver execute\r\n\r\nDo not wait after last possible attempt in driver execute_with_retry\r\n\r\n* Fix naming of max attempts in driver\r\n\r\n* Add max attempt to lsf test\r\n\r\nRemove retries due to stdout missing",
          "timestamp": "2024-11-26T15:21:17+01:00",
          "tree_id": "a2a5066c6e3cf411c3ec7d8cd65fa875de5deaa6",
          "url": "https://github.com/equinor/ert/commit/ca1806e59d1a6a45a18cbc4a3c4ca51af6b4db0b"
        },
        "date": 1732630987772,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19233637591227304,
            "unit": "iter/sec",
            "range": "stddev: 0.04663167774807077",
            "extra": "mean: 5.199224511 sec\nrounds: 5"
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
          "id": "398aa3e77980a329bc9a91e5a540fb2cc99ea03e",
          "message": "Unpin hypothesis",
          "timestamp": "2024-11-27T09:15:00+01:00",
          "tree_id": "c2d72160ec8103e25d449d1fb65a240b51acf709",
          "url": "https://github.com/equinor/ert/commit/398aa3e77980a329bc9a91e5a540fb2cc99ea03e"
        },
        "date": 1732695417284,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1942073480123078,
            "unit": "iter/sec",
            "range": "stddev: 0.035025050932171516",
            "extra": "mean: 5.149135757400001 sec\nrounds: 5"
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
          "id": "dd7925c48926cce52e2e0cadb3994ec11658cf45",
          "message": "Remove the everest load entry point",
          "timestamp": "2024-11-27T09:23:34+01:00",
          "tree_id": "af2ddaa3ce8c3b0378506b91930c215d5ca470b3",
          "url": "https://github.com/equinor/ert/commit/dd7925c48926cce52e2e0cadb3994ec11658cf45"
        },
        "date": 1732695922113,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1922525504311773,
            "unit": "iter/sec",
            "range": "stddev: 0.02507621422219319",
            "extra": "mean: 5.201491464000009 sec\nrounds: 5"
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
          "id": "447d647c224ffc7f94fcd905d4597aff55a766f4",
          "message": "Use explicit everest config components when creating ropt config object",
          "timestamp": "2024-11-27T17:40:26+09:00",
          "tree_id": "eb504b41594721a723b792d04ac07834cf0bf29b",
          "url": "https://github.com/equinor/ert/commit/447d647c224ffc7f94fcd905d4597aff55a766f4"
        },
        "date": 1732696933729,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19115550773424858,
            "unit": "iter/sec",
            "range": "stddev: 0.029059871232901543",
            "extra": "mean: 5.231342857199996 sec\nrounds: 5"
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
          "id": "28c0abd02071268c47353ad4b04c9163d188f900",
          "message": "Add '-sv' flag when running everest integration tests",
          "timestamp": "2024-11-27T18:34:44+09:00",
          "tree_id": "f79c6fd84430f9ab44519d88964e63f223652804",
          "url": "https://github.com/equinor/ert/commit/28c0abd02071268c47353ad4b04c9163d188f900"
        },
        "date": 1732700198161,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19072920572991123,
            "unit": "iter/sec",
            "range": "stddev: 0.064560775861534",
            "extra": "mean: 5.243035518200003 sec\nrounds: 5"
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
          "id": "25e431fb013c3ae78b10a3fb95485df8292e586b",
          "message": "Inject plugin step config into jobs-json for steps\n\nGeneral configuration using key-values via the plugin system for\nindividual steps will be merged with environment property of each\nForwardModelStep that is dumped as json in every runpath.",
          "timestamp": "2024-11-27T12:56:58+01:00",
          "tree_id": "db45877f059d3bc6ce1a35a11035871aa3e4e697",
          "url": "https://github.com/equinor/ert/commit/25e431fb013c3ae78b10a3fb95485df8292e586b"
        },
        "date": 1732708732581,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19663108550829128,
            "unit": "iter/sec",
            "range": "stddev: 0.03254399607322415",
            "extra": "mean: 5.085665867199992 sec\nrounds: 5"
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
          "id": "9a89ae3b0d33da721d942ba604bcd17684be473e",
          "message": "Have fm_runner's event reporter shutdown gracefully\n\nThis commit fixes the issue where the logs would be spammed with errors\nrelated to the websocket client being forcefully shut down before\nclosing the connection.\nIt also fixes the issue where the fm_runner was not killing the running\nforward models when sigterm was signaled",
          "timestamp": "2024-11-27T15:41:26+01:00",
          "tree_id": "b4a936adfe1c2c401cdfe541060211ad55ca6027",
          "url": "https://github.com/equinor/ert/commit/9a89ae3b0d33da721d942ba604bcd17684be473e"
        },
        "date": 1732718601058,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1898650602845553,
            "unit": "iter/sec",
            "range": "stddev: 0.03156183523647813",
            "extra": "mean: 5.266898493600013 sec\nrounds: 5"
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
          "id": "7e0f7931d1cd8f8105d250e59bc7be24c9eddaa8",
          "message": "Select first realizations by default in gui\n\n* Select first realizations by default in gui\r\n\r\nCo-authored-by: Andreas Eknes Lie <andrli@equinor.com>",
          "timestamp": "2024-11-27T15:56:41+01:00",
          "tree_id": "fdb807023a605162266a8b86aed692f0a3bf9467",
          "url": "https://github.com/equinor/ert/commit/7e0f7931d1cd8f8105d250e59bc7be24c9eddaa8"
        },
        "date": 1732719513642,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1921977827711131,
            "unit": "iter/sec",
            "range": "stddev: 0.032388740798183685",
            "extra": "mean: 5.202973653400011 sec\nrounds: 5"
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
          "id": "403a55e5348ff3e0b3f3e4b28cfa1ac2127562f8",
          "message": "Make sure driver is polled for status",
          "timestamp": "2024-11-27T20:28:56+01:00",
          "tree_id": "a249a41a37dc21fc4469373e89369e91363a7bb5",
          "url": "https://github.com/equinor/ert/commit/403a55e5348ff3e0b3f3e4b28cfa1ac2127562f8"
        },
        "date": 1732735847739,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18800103317336436,
            "unit": "iter/sec",
            "range": "stddev: 0.012209625654044398",
            "extra": "mean: 5.319119704399998 sec\nrounds: 5"
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
          "id": "2d3301d3bd93544902f536b99aad29a2b1f222b7",
          "message": "Log that NUM_CPU was parsed from DATA",
          "timestamp": "2024-11-28T08:41:45+01:00",
          "tree_id": "e85aa4b36b558686e2b63e3aeeae3ade3eb81adb",
          "url": "https://github.com/equinor/ert/commit/2d3301d3bd93544902f536b99aad29a2b1f222b7"
        },
        "date": 1732779814313,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19545313991014893,
            "unit": "iter/sec",
            "range": "stddev: 0.021148194854071628",
            "extra": "mean: 5.116315861999999 sec\nrounds: 5"
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
          "id": "2543b2c5e47acf276fa1762dc7d1af4014902c84",
          "message": "Fix test when other plugins could be installed",
          "timestamp": "2024-11-28T09:00:49+01:00",
          "tree_id": "d3ae72a8cab44648765fe123a4be94767d9014bd",
          "url": "https://github.com/equinor/ert/commit/2543b2c5e47acf276fa1762dc7d1af4014902c84"
        },
        "date": 1732780964111,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19378326471071458,
            "unit": "iter/sec",
            "range": "stddev: 0.03026455207052366",
            "extra": "mean: 5.160404338799998 sec\nrounds: 5"
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
          "id": "dfffa695ea16d40c8f3817c097d720060deade8e",
          "message": "Fix manifest having too many files and not substituting\n\nAlso makes substitution of <IENS> and <ITER> for\r\nall parameters and responses",
          "timestamp": "2024-11-28T08:29:12Z",
          "tree_id": "06a8bdaca3463bb2246ae982c61b63f28bd9560a",
          "url": "https://github.com/equinor/ert/commit/dfffa695ea16d40c8f3817c097d720060deade8e"
        },
        "date": 1732782658775,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1950929591724109,
            "unit": "iter/sec",
            "range": "stddev: 0.04992714006316286",
            "extra": "mean: 5.125761607400003 sec\nrounds: 5"
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
          "id": "b556d87540a2064534233622246473533b6d6773",
          "message": "Mention runpath in copy_file docs",
          "timestamp": "2024-11-28T09:32:34+01:00",
          "tree_id": "63b9725924970e16005d095388657f894c95e67d",
          "url": "https://github.com/equinor/ert/commit/b556d87540a2064534233622246473533b6d6773"
        },
        "date": 1732782863397,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19532027537208577,
            "unit": "iter/sec",
            "range": "stddev: 0.018070202769042913",
            "extra": "mean: 5.119796181399994 sec\nrounds: 5"
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
          "id": "8848f373b965f02af91bb185c1afabf90d5897f6",
          "message": "Avoid using .response_configuration within loops",
          "timestamp": "2024-11-28T09:34:26+01:00",
          "tree_id": "e67fbe7463f11ea2e5d68060e0a91cd540018b4e",
          "url": "https://github.com/equinor/ert/commit/8848f373b965f02af91bb185c1afabf90d5897f6"
        },
        "date": 1732782972499,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1946136955328512,
            "unit": "iter/sec",
            "range": "stddev: 0.049536419705268814",
            "extra": "mean: 5.138384517399999 sec\nrounds: 5"
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
          "id": "68c2de998a110df4c297353256982b487ddab767",
          "message": "Add time&mem test for snake oil plotting",
          "timestamp": "2024-11-28T09:34:45+01:00",
          "tree_id": "9ef010a445c0afb2780e07e62c392418c025ffff",
          "url": "https://github.com/equinor/ert/commit/68c2de998a110df4c297353256982b487ddab767"
        },
        "date": 1732783001315,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1923963015343426,
            "unit": "iter/sec",
            "range": "stddev: 0.02450197563491844",
            "extra": "mean: 5.1976051100000005 sec\nrounds: 5"
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
          "id": "53eea27b8b6c7a004ecf1ce485cb7ef1b5683f64",
          "message": "Remove realizations_initialized\n\nused in test only",
          "timestamp": "2024-11-28T09:44:26+01:00",
          "tree_id": "2eaaa7daa6c7e385f66841349c2c75b11e2d3b76",
          "url": "https://github.com/equinor/ert/commit/53eea27b8b6c7a004ecf1ce485cb7ef1b5683f64"
        },
        "date": 1732783579492,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19410179268395863,
            "unit": "iter/sec",
            "range": "stddev: 0.04247093091351129",
            "extra": "mean: 5.151935931000002 sec\nrounds: 5"
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
          "id": "9c0df14d0288e9827ca4aab20fcc240255a82191",
          "message": "Only get processtree once per report",
          "timestamp": "2024-11-28T10:33:02+01:00",
          "tree_id": "3a1c9c1eccf762eea8a18eb2b1672ce2a90dfb72",
          "url": "https://github.com/equinor/ert/commit/9c0df14d0288e9827ca4aab20fcc240255a82191"
        },
        "date": 1732786495679,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1954387200893491,
            "unit": "iter/sec",
            "range": "stddev: 0.03417556641816042",
            "extra": "mean: 5.116693353 sec\nrounds: 5"
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
          "id": "af3600938b9d49683035a1d32eed43af53131cf7",
          "message": "Ensure tests run in clean directory\n\nNot doing this causes tests to step on each others toes, yielding\nflakyness",
          "timestamp": "2024-11-28T12:04:39+01:00",
          "tree_id": "0d2f0d1bf861c30c163c70cf62fbb17b5f6a3619",
          "url": "https://github.com/equinor/ert/commit/af3600938b9d49683035a1d32eed43af53131cf7"
        },
        "date": 1732791989925,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19438592104776742,
            "unit": "iter/sec",
            "range": "stddev: 0.0398985068486518",
            "extra": "mean: 5.144405493000005 sec\nrounds: 5"
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
          "id": "c3b10c4b775dc1cbffaac11b13cd27ea510c28fa",
          "message": "Simplify ensemble evaluator shutdown timeout",
          "timestamp": "2024-11-28T13:53:44+01:00",
          "tree_id": "dddbb72c8d9a85209b7a63881b8ffe921fee594f",
          "url": "https://github.com/equinor/ert/commit/c3b10c4b775dc1cbffaac11b13cd27ea510c28fa"
        },
        "date": 1732798536499,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18949106549432843,
            "unit": "iter/sec",
            "range": "stddev: 0.04941772022760424",
            "extra": "mean: 5.27729366760002 sec\nrounds: 5"
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
          "id": "55793cd3b175be581cc87b4c741fca77914854db",
          "message": "Enforce correct dtypes in migration",
          "timestamp": "2024-11-28T13:56:16+01:00",
          "tree_id": "c67fff1ed97ea571d423e5be48fa763919ed0f83",
          "url": "https://github.com/equinor/ert/commit/55793cd3b175be581cc87b4c741fca77914854db"
        },
        "date": 1732798696062,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18929199575224825,
            "unit": "iter/sec",
            "range": "stddev: 0.06980627750871395",
            "extra": "mean: 5.282843556199987 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}