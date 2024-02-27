window.BENCHMARK_DATA = {
  "lastUpdate": 1709040411219,
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
          "id": "9305dca433d6a7457172cc62a06e452077559c1d",
          "message": "Specify diffing per file extension",
          "timestamp": "2024-02-22T14:51:20+01:00",
          "tree_id": "5e62b2f32e995b3d2a91f8f02e898f499850cbd0",
          "url": "https://github.com/equinor/ert/commit/9305dca433d6a7457172cc62a06e452077559c1d"
        },
        "date": 1708610045314,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.673812391748218,
            "unit": "iter/sec",
            "range": "stddev: 0.03353848941566597",
            "extra": "mean: 149.83939333332805 msec\nrounds: 6"
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
          "id": "9ad807253b3183cda424aac6e22f9ec1b5ad4579",
          "message": "Merge read_mask and get_mask",
          "timestamp": "2024-02-22T14:56:37+01:00",
          "tree_id": "d2f42c42916c3e3d0e65e949a59a8efad7cfc6e0",
          "url": "https://github.com/equinor/ert/commit/9ad807253b3183cda424aac6e22f9ec1b5ad4579"
        },
        "date": 1708610347717,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.506934207161687,
            "unit": "iter/sec",
            "range": "stddev: 0.033103392838769376",
            "extra": "mean: 153.6822054999997 msec\nrounds: 6"
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
          "id": "83cb732cf3a29f4d321b712b5fd789f93230589b",
          "message": "Replace deprecated api calls to pydantic\n\nReplaced usage of Config class with model_fields or dataclasses\nReplaced usage of .dict() with .model_dump()",
          "timestamp": "2024-02-23T09:08:08+01:00",
          "tree_id": "262ced5b1b7a360c9004e7eb6643946686a0aa7b",
          "url": "https://github.com/equinor/ert/commit/83cb732cf3a29f4d321b712b5fd789f93230589b"
        },
        "date": 1708675851277,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.464613442443334,
            "unit": "iter/sec",
            "range": "stddev: 0.03772270472868877",
            "extra": "mean: 154.68829016666538 msec\nrounds: 6"
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
          "id": "b3457ad5d71ea24af9c76c7d7f3e64e7f355a548",
          "message": "Remove incorrect documentation of lsf_server",
          "timestamp": "2024-02-23T09:10:27+01:00",
          "tree_id": "8ee6deaab9674367eeadb0e37177c8fd39cd46c9",
          "url": "https://github.com/equinor/ert/commit/b3457ad5d71ea24af9c76c7d7f3e64e7f355a548"
        },
        "date": 1708676008416,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.574738157289589,
            "unit": "iter/sec",
            "range": "stddev: 0.10950988472155422",
            "extra": "mean: 179.38062233333577 msec\nrounds: 6"
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
          "id": "19cc1b42e99266c84c93db18a7c30d2dbb1399e3",
          "message": "Add property to check storage availability\n\nAvoids having to call notifiers private _storage",
          "timestamp": "2024-02-23T09:32:31+01:00",
          "tree_id": "c1717c0ea4bfdb97f26c51e2b93075e3b84ba7c2",
          "url": "https://github.com/equinor/ert/commit/19cc1b42e99266c84c93db18a7c30d2dbb1399e3"
        },
        "date": 1708677320538,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.621359331235858,
            "unit": "iter/sec",
            "range": "stddev: 0.03262994424865329",
            "extra": "mean: 151.0263904999931 msec\nrounds: 6"
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
          "id": "a7682a733a82f4a5f17c2e74f14b378e93d36520",
          "message": "Documented target_case_format\n\nAttempted to make it more clear that what is done is fall through\noptions.",
          "timestamp": "2024-02-23T09:41:15+01:00",
          "tree_id": "2a6fafac137dfcf875d5985c950f628f276c432b",
          "url": "https://github.com/equinor/ert/commit/a7682a733a82f4a5f17c2e74f14b378e93d36520"
        },
        "date": 1708677843379,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.4978207018713565,
            "unit": "iter/sec",
            "range": "stddev: 0.039246160741786905",
            "extra": "mean: 153.89775216666143 msec\nrounds: 6"
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
          "id": "78a43e3b38136efdb6c1b568988acbb15bb4bba4",
          "message": "Detemine source for update in separate function",
          "timestamp": "2024-02-23T10:23:43+01:00",
          "tree_id": "6ee972002c027e412fcc936bcda94ecc0694d97b",
          "url": "https://github.com/equinor/ert/commit/78a43e3b38136efdb6c1b568988acbb15bb4bba4"
        },
        "date": 1708680371101,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.670911122814243,
            "unit": "iter/sec",
            "range": "stddev: 0.02979135776120993",
            "extra": "mean: 149.90456049999543 msec\nrounds: 6"
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
          "id": "06ac1ae01cdc1d4fc54e987789c3ccb0af820d3b",
          "message": "Extract determining restart info into function",
          "timestamp": "2024-02-23T10:24:35+01:00",
          "tree_id": "cadbb0005d4071145b869708dffde3c264544faf",
          "url": "https://github.com/equinor/ert/commit/06ac1ae01cdc1d4fc54e987789c3ccb0af820d3b"
        },
        "date": 1708680450838,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.235065265078,
            "unit": "iter/sec",
            "range": "stddev: 0.030417087324497136",
            "extra": "mean: 160.3832450000008 msec\nrounds: 6"
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
          "id": "69f03eb22f170e07cb4199fab940cbe3e439f07a",
          "message": "Remove unused _active_realizations_model",
          "timestamp": "2024-02-23T14:32:32+01:00",
          "tree_id": "e1b684d25f261e9dbf15f2b709f61471887fbc42",
          "url": "https://github.com/equinor/ert/commit/69f03eb22f170e07cb4199fab940cbe3e439f07a"
        },
        "date": 1708695319916,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.541445740556937,
            "unit": "iter/sec",
            "range": "stddev: 0.0368734203831908",
            "extra": "mean: 152.87140483333891 msec\nrounds: 6"
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
          "id": "583022faa978c55e1aed7ed4798201748f71415d",
          "message": "Extract batch-size calculation to function",
          "timestamp": "2024-02-23T14:35:33+01:00",
          "tree_id": "82bf979c261c94bea09cc8c13adab92b7d4b22f6",
          "url": "https://github.com/equinor/ert/commit/583022faa978c55e1aed7ed4798201748f71415d"
        },
        "date": 1708695485983,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.71701753459302,
            "unit": "iter/sec",
            "range": "stddev: 0.03357219737840208",
            "extra": "mean: 148.87559766666433 msec\nrounds: 6"
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
          "id": "ee8b423a41a75492da92fcc637bbf61abaf15bfe",
          "message": "Replace gstools with scipy\n\nTo get a more realistic test, we want to run\r\nlocalization on more parameter.\r\ngstools is very slow so generating many parameters would take\r\na lot of time.\r\nInstead, we generate a gaussian field using scipy which is\r\nless flexible but much faster.",
          "timestamp": "2024-02-23T14:52:03+01:00",
          "tree_id": "d036b749117e7c74b14e5a7887a17a86fcecadfd",
          "url": "https://github.com/equinor/ert/commit/ee8b423a41a75492da92fcc637bbf61abaf15bfe"
        },
        "date": 1708696495932,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4572465609555191,
            "unit": "iter/sec",
            "range": "stddev: 0.5187679115716413",
            "extra": "mean: 2.187003873600003 sec\nrounds: 5"
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
          "id": "21f0a89677b45fcfd4367e262b9f14f6d5a09230",
          "message": "Set opm-flow path depending on rhel version in tests",
          "timestamp": "2024-02-23T15:33:57+01:00",
          "tree_id": "f1e377f028adaabe2d7c2efad80d7abec4d011e6",
          "url": "https://github.com/equinor/ert/commit/21f0a89677b45fcfd4367e262b9f14f6d5a09230"
        },
        "date": 1708699022381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4457278465810794,
            "unit": "iter/sec",
            "range": "stddev: 0.5389480966299462",
            "extra": "mean: 2.243521484400003 sec\nrounds: 5"
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
          "id": "2920d31f8cd8b68f71caa1b5521f6392da65348a",
          "message": "Refactor SingleTestRun run_experiment",
          "timestamp": "2024-02-26T10:07:41+01:00",
          "tree_id": "fe06dd565c0c5e66849d93b276c09854fe690741",
          "url": "https://github.com/equinor/ert/commit/2920d31f8cd8b68f71caa1b5521f6392da65348a"
        },
        "date": 1708938631930,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.45168905271970416,
            "unit": "iter/sec",
            "range": "stddev: 0.5466795117190915",
            "extra": "mean: 2.2139124116000004 sec\nrounds: 5"
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
          "id": "322984cb20961bca3a81bae8d12a2352f9428254",
          "message": "Fix default in _iterative_case_format\n\na7682a733a82f4a5f17c2e74f14b378e93d36520 introduced an unintended\nchange where the case format could become \"None_%d\".",
          "timestamp": "2024-02-26T10:30:15+01:00",
          "tree_id": "3047e303331e66f25844636b82fb9dfcdb932e21",
          "url": "https://github.com/equinor/ert/commit/322984cb20961bca3a81bae8d12a2352f9428254"
        },
        "date": 1708939998761,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4679230891682248,
            "unit": "iter/sec",
            "range": "stddev: 0.44534930662652117",
            "extra": "mean: 2.1371033470000156 sec\nrounds: 5"
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
          "id": "b9f3f12ea6a8cf8f4a4c44a1e2efb5585d97908e",
          "message": "Use .[dev] when installing ert",
          "timestamp": "2024-02-26T12:09:08+01:00",
          "tree_id": "6a665a53695d3a1e6e5889b48b574018e3da9577",
          "url": "https://github.com/equinor/ert/commit/b9f3f12ea6a8cf8f4a4c44a1e2efb5585d97908e"
        },
        "date": 1708945928843,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4259933197872894,
            "unit": "iter/sec",
            "range": "stddev: 0.49907934152606415",
            "extra": "mean: 2.3474546513999996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sonso@equinor.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "committer": {
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "7ed11b100242112bb11ed88d5089ebaab21c4c8c",
          "message": "Use Trusted publishing for PyPi",
          "timestamp": "2024-02-26T12:59:57+01:00",
          "tree_id": "d0d79fc872cf42666d2e5b9a3fa00da42dd96156",
          "url": "https://github.com/equinor/ert/commit/7ed11b100242112bb11ed88d5089ebaab21c4c8c"
        },
        "date": 1708948968987,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.44467054657133825,
            "unit": "iter/sec",
            "range": "stddev: 0.4318955426984798",
            "extra": "mean: 2.2488559399999986 sec\nrounds: 5"
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
          "id": "4b1dcf7d87b60efe45e0772805f6361021bc1e3c",
          "message": "Catch more errors from reading datasets\n\nWe now raise more error types when reading datasets, so will start\ncatching broader exceptions.",
          "timestamp": "2024-02-26T13:12:35+01:00",
          "tree_id": "8e8530953d445484e031c557dfe0e93e22136835",
          "url": "https://github.com/equinor/ert/commit/4b1dcf7d87b60efe45e0772805f6361021bc1e3c"
        },
        "date": 1708949737240,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4664097025071636,
            "unit": "iter/sec",
            "range": "stddev: 0.5015753909263939",
            "extra": "mean: 2.1440377304000036 sec\nrounds: 5"
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
          "id": "43a5885cef048c2bf0ae63e1b488c0866dc1e5e2",
          "message": "Downgrade logging levels in OpenPBS driver\n\nTry to avoid log levels at INFO and above for things that might happen\nfor every realization in an ensemble.",
          "timestamp": "2024-02-26T13:37:30+01:00",
          "tree_id": "f01be24cc3e7926ce7d9d2d3be5b3b416d8f411f",
          "url": "https://github.com/equinor/ert/commit/43a5885cef048c2bf0ae63e1b488c0866dc1e5e2"
        },
        "date": 1708951228381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4699301959549671,
            "unit": "iter/sec",
            "range": "stddev: 0.4701161233962551",
            "extra": "mean: 2.1279756197999857 sec\nrounds: 5"
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
          "id": "fe12aea68ff70831950861af75fc3bcf3202a97a",
          "message": "Refactor FeatureToggling to FeatureScheduler\n\nWe only need to toggle scheduler. If we ever need to toggle some other\nfeature, we can implement that as a separate class when the time comes.\nThis commit removes the complexity associated with supporting multiple\nsuch features.",
          "timestamp": "2024-02-26T13:59:23+01:00",
          "tree_id": "ef71e0f808b76f4d69acb51d46024a8e1b95f819",
          "url": "https://github.com/equinor/ert/commit/fe12aea68ff70831950861af75fc3bcf3202a97a"
        },
        "date": 1708952584349,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.48470901696729174,
            "unit": "iter/sec",
            "range": "stddev: 0.4832250150296217",
            "extra": "mean: 2.063093454000011 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "107626001+jonathan-eq@users.noreply.github.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df902d1a08777ded886c08ee78ea28067ab34b56",
          "message": "Increase gui test timeout period (#7274)",
          "timestamp": "2024-02-26T15:18:26+01:00",
          "tree_id": "68362d56e7a6068abbd047f2d38a81a5fcecaef7",
          "url": "https://github.com/equinor/ert/commit/df902d1a08777ded886c08ee78ea28067ab34b56"
        },
        "date": 1708957275390,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4484638111181366,
            "unit": "iter/sec",
            "range": "stddev: 0.4218459239962214",
            "extra": "mean: 2.2298343259999966 sec\nrounds: 5"
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
          "id": "9c6e59595c692ac5affff179152cbd8b6c4a9820",
          "message": "Gather observation errors instead of failing on first one\n\nAdd location context to observation errors when possible",
          "timestamp": "2024-02-27T09:29:21+01:00",
          "tree_id": "4613ccc20a2a3e80f0b69a0d8b2cc678a1e56ddb",
          "url": "https://github.com/equinor/ert/commit/9c6e59595c692ac5affff179152cbd8b6c4a9820"
        },
        "date": 1709022741469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.45194113884368653,
            "unit": "iter/sec",
            "range": "stddev: 0.5568441341634356",
            "extra": "mean: 2.212677523799999 sec\nrounds: 5"
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
          "id": "bb253179549ed2b758ce34d5ace5fe015866253b",
          "message": "Fix bug where experiment name always became the last observation name",
          "timestamp": "2024-02-27T09:38:56+01:00",
          "tree_id": "989da6d407ac59a017456242dd9f325393d3548f",
          "url": "https://github.com/equinor/ert/commit/bb253179549ed2b758ce34d5ace5fe015866253b"
        },
        "date": 1709023313015,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.43309850875977257,
            "unit": "iter/sec",
            "range": "stddev: 0.49289516813413886",
            "extra": "mean: 2.308943530799991 sec\nrounds: 5"
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
          "id": "c547f8d1bdca1ba221086b0f0481e915cfe39f1c",
          "message": "Cache experiment observations",
          "timestamp": "2024-02-27T10:09:05+01:00",
          "tree_id": "ced3d3e78c87eaac91fd20b51435dde7c6e0d22f",
          "url": "https://github.com/equinor/ert/commit/c547f8d1bdca1ba221086b0f0481e915cfe39f1c"
        },
        "date": 1709025117584,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4547075649023158,
            "unit": "iter/sec",
            "range": "stddev: 0.47466350184896633",
            "extra": "mean: 2.199215665599996 sec\nrounds: 5"
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
          "id": "ce1cc0b9dcc8571e28ab3fee41fd70dbdfed6c15",
          "message": "Forward exceptions from threads to main thread\n\nWe substitute usages of `threading.Thread` with `ErtThread`. These are\nmostly equivalent, with the exception that `ErtThread` catches all\nexceptions that are propagated to the `run` function. This exception is\nthen saved globally before we raise a `SIGUSR1` signal to ourselves so\nthat the main thread can wake up and raise the signal from its context.\nIn practice, this means that any part of Ert can throw an exception and\nhave it be propagated to `cli.main` to be processed there.\n\nIt should also be noted that returning from `SIGUSR1` handler is\npermitted, and so eg. Ert GUI could continue rather than quit.",
          "timestamp": "2024-02-27T10:11:03+01:00",
          "tree_id": "62b8417c9345533de45c1c6ae77253af348fa4c6",
          "url": "https://github.com/equinor/ert/commit/ce1cc0b9dcc8571e28ab3fee41fd70dbdfed6c15"
        },
        "date": 1709025238660,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4738080601976955,
            "unit": "iter/sec",
            "range": "stddev: 0.4319376865916574",
            "extra": "mean: 2.110559283400016 sec\nrounds: 5"
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
          "id": "7720ee435f04af4196fe6ac5d2145b986cba8666",
          "message": "Revert \"Forward exceptions from threads to main thread\"\n\nThis reverts commit ce1cc0b9dcc8571e28ab3fee41fd70dbdfed6c15.",
          "timestamp": "2024-02-27T12:11:53+01:00",
          "tree_id": "ced3d3e78c87eaac91fd20b51435dde7c6e0d22f",
          "url": "https://github.com/equinor/ert/commit/7720ee435f04af4196fe6ac5d2145b986cba8666"
        },
        "date": 1709032490644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.462430219628993,
            "unit": "iter/sec",
            "range": "stddev: 0.568056212471504",
            "extra": "mean: 2.162488430799999 sec\nrounds: 5"
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
          "id": "8c181d0257f516818ccf377db1198d52d85b3eef",
          "message": "Support NUM_NODES and NUM_CPUS_PER_NODE in OpenPBS driver",
          "timestamp": "2024-02-27T13:32:36+01:00",
          "tree_id": "cc72bbb5e8e26831133c88ba61ab4cd1719f551f",
          "url": "https://github.com/equinor/ert/commit/8c181d0257f516818ccf377db1198d52d85b3eef"
        },
        "date": 1709037343515,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4645685465735202,
            "unit": "iter/sec",
            "range": "stddev: 0.45524585472737855",
            "extra": "mean: 2.152534878599977 sec\nrounds: 5"
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
          "id": "29f2cf2b8b5a6d5109e58bfb16b36f4f0258a0ce",
          "message": "Remove resdata as a build dependency",
          "timestamp": "2024-02-27T13:35:14+01:00",
          "tree_id": "02218811fa30130dec6beefa7dbe2dd7d230b71b",
          "url": "https://github.com/equinor/ert/commit/29f2cf2b8b5a6d5109e58bfb16b36f4f0258a0ce"
        },
        "date": 1709037539843,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.452558182289641,
            "unit": "iter/sec",
            "range": "stddev: 0.36902955692234224",
            "extra": "mean: 2.2096606339999654 sec\nrounds: 5"
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
          "id": "03609d91f48e964e2dad132ff03412c874db9a39",
          "message": "Tune error handling in Python LSF driver for bsub\n\nNo known error scenarios for which the bsub command\nshould be retried is currently known.\n\nAdd tests for the behaviour in error conditions",
          "timestamp": "2024-02-27T13:59:13+01:00",
          "tree_id": "7303c1501d311d91abff5cd04d598eb684bfbad3",
          "url": "https://github.com/equinor/ert/commit/03609d91f48e964e2dad132ff03412c874db9a39"
        },
        "date": 1709038934123,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.44561497901876196,
            "unit": "iter/sec",
            "range": "stddev: 0.47437871720607133",
            "extra": "mean: 2.244089734600004 sec\nrounds: 5"
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
          "id": "d122ab3616d0d571ee10c5bb6acd0d46ca7da059",
          "message": "Add test function combining max_submit and max_runtime",
          "timestamp": "2024-02-27T14:02:05+01:00",
          "tree_id": "e42d687726448b6389004c13cb23c920fa58a1e3",
          "url": "https://github.com/equinor/ert/commit/d122ab3616d0d571ee10c5bb6acd0d46ca7da059"
        },
        "date": 1709039103467,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.47159243490411523,
            "unit": "iter/sec",
            "range": "stddev: 0.5422954022475468",
            "extra": "mean: 2.1204750670000068 sec\nrounds: 5"
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
          "id": "29ff4fe7bef05429784f0e5834636e1b34c6b729",
          "message": "Replace use of TypedDict with dataclasses in observation parser",
          "timestamp": "2024-02-27T14:23:53+01:00",
          "tree_id": "a74845e255afab4fe6fe4e9ba6dd85c3bdae4d5b",
          "url": "https://github.com/equinor/ert/commit/29ff4fe7bef05429784f0e5834636e1b34c6b729"
        },
        "date": 1709040410701,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.45048805755875826,
            "unit": "iter/sec",
            "range": "stddev: 0.4402250864653936",
            "extra": "mean: 2.219814672599989 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}