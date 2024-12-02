window.BENCHMARK_DATA = {
  "lastUpdate": 1733142466097,
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
          "id": "f6b7bd63554a9d9fe34cd479dca9869705262910",
          "message": "Make sure site config is loaded for Everest",
          "timestamp": "2024-11-28T15:05:14+01:00",
          "tree_id": "83b0de76f9a2c4ad440ce377b6146fedbe2bdf78",
          "url": "https://github.com/equinor/ert/commit/f6b7bd63554a9d9fe34cd479dca9869705262910"
        },
        "date": 1732802823848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19147649349749385,
            "unit": "iter/sec",
            "range": "stddev: 0.03555141279673454",
            "extra": "mean: 5.222573182400003 sec\nrounds: 5"
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
          "id": "d176350d66a30f633d211d8a7108b35826c80392",
          "message": "Suppress everest logger output to the console (#9371)\n\nSuppress everest logger output to the console",
          "timestamp": "2024-11-29T08:15:33+01:00",
          "tree_id": "76df32021bf81680ae86d8a335f262781c8dd4f9",
          "url": "https://github.com/equinor/ert/commit/d176350d66a30f633d211d8a7108b35826c80392"
        },
        "date": 1732864642264,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1968155059439951,
            "unit": "iter/sec",
            "range": "stddev: 0.02789146693841389",
            "extra": "mean: 5.08090048700002 sec\nrounds: 5"
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
          "id": "1df489a295c252968d777f973eb8be911f9ed0e1",
          "message": "Remove lambda function",
          "timestamp": "2024-11-29T09:45:49+01:00",
          "tree_id": "f140d3e959289822806f63ddcf95ac707cb819fd",
          "url": "https://github.com/equinor/ert/commit/1df489a295c252968d777f973eb8be911f9ed0e1"
        },
        "date": 1732870064072,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19077687544305852,
            "unit": "iter/sec",
            "range": "stddev: 0.07491276480983348",
            "extra": "mean: 5.2417254328000125 sec\nrounds: 5"
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
          "id": "f39a174f5e8559503f57129f7e08c9f95196bf44",
          "message": "Replace everest site config manager with ert",
          "timestamp": "2024-11-29T09:46:27+01:00",
          "tree_id": "b53ebf3b45565e26d3de33eab288783e9ac017af",
          "url": "https://github.com/equinor/ert/commit/f39a174f5e8559503f57129f7e08c9f95196bf44"
        },
        "date": 1732870098320,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19133164376739084,
            "unit": "iter/sec",
            "range": "stddev: 0.014603867025298803",
            "extra": "mean: 5.22652698899999 sec\nrounds: 5"
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
          "id": "a99ed165e9903d8ee05caef418f90796ef9eb458",
          "message": "Remove unused function",
          "timestamp": "2024-11-29T12:05:34+01:00",
          "tree_id": "c7e27794fbc4dbc87a3bbf18e5df30eaf2f28750",
          "url": "https://github.com/equinor/ert/commit/a99ed165e9903d8ee05caef418f90796ef9eb458"
        },
        "date": 1732878444313,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19727052440242182,
            "unit": "iter/sec",
            "range": "stddev: 0.04723139479139347",
            "extra": "mean: 5.0691810295999975 sec\nrounds: 5"
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
          "id": "105e41a754091570d234a7e251f88acf40b9c9f4",
          "message": "Make sure that migrations use correct dtypes",
          "timestamp": "2024-11-29T12:20:40+01:00",
          "tree_id": "eb258b2fc42c35cb54c73f5be5d8b40cd39f543c",
          "url": "https://github.com/equinor/ert/commit/105e41a754091570d234a7e251f88acf40b9c9f4"
        },
        "date": 1732879350277,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1923995306790705,
            "unit": "iter/sec",
            "range": "stddev: 0.03137511752879723",
            "extra": "mean: 5.197517875800003 sec\nrounds: 5"
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
          "id": "6197ed4ebca19c46cd44b1080c6ce60fb9f59222",
          "message": "Reduce examples ran in storage test",
          "timestamp": "2024-12-02T09:32:16+01:00",
          "tree_id": "7492c002fb1833ea645140662fdadabb22a49d8c",
          "url": "https://github.com/equinor/ert/commit/6197ed4ebca19c46cd44b1080c6ce60fb9f59222"
        },
        "date": 1733128453040,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19529733591711776,
            "unit": "iter/sec",
            "range": "stddev: 0.02977841711705624",
            "extra": "mean: 5.120397548199992 sec\nrounds: 5"
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
      }
    ]
  }
}