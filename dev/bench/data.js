window.BENCHMARK_DATA = {
  "lastUpdate": 1708949737916,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "61c4e6e04f471f67be6a2a496ed3b319e0514944",
          "message": "Use macOS-13-xl ARM runners for macOS build\n\nAlso increase build and testing with python versions 3.11 & 3.12",
          "timestamp": "2024-02-16T12:22:28+01:00",
          "tree_id": "70a6f2951e9c1b4425349aaadefdf7c8d609ba3e",
          "url": "https://github.com/equinor/ert/commit/61c4e6e04f471f67be6a2a496ed3b319e0514944"
        },
        "date": 1708082709605,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.414432281054572,
            "unit": "iter/sec",
            "range": "stddev: 0.030852809887946295",
            "extra": "mean: 155.89844216666884 msec\nrounds: 6"
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
          "id": "e3ef768cc300f8aa410eac0c4584b65a4604af05",
          "message": "Avoid generating the starting date in observations",
          "timestamp": "2024-02-16T13:56:36+01:00",
          "tree_id": "bb6c545fb36dd18b52250313956cbc949020517f",
          "url": "https://github.com/equinor/ert/commit/e3ef768cc300f8aa410eac0c4584b65a4604af05"
        },
        "date": 1708088377169,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.38325260568501,
            "unit": "iter/sec",
            "range": "stddev: 0.030421585609813433",
            "extra": "mean: 156.65994466667144 msec\nrounds: 6"
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
          "id": "037c0b22ca6d0232ca1d65c6b560e1168479b1b9",
          "message": "Revert \"Skip export misfit data job test on mac\"\n\nThis reverts commit 4cc9d21f9ff6d613712b5681420ea20abc2e81d9.",
          "timestamp": "2024-02-19T09:00:36+01:00",
          "tree_id": "f3b3c5ef6f30c95316224281887d66251cb2f834",
          "url": "https://github.com/equinor/ert/commit/037c0b22ca6d0232ca1d65c6b560e1168479b1b9"
        },
        "date": 1708329798532,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.500515484377715,
            "unit": "iter/sec",
            "range": "stddev: 0.03602155084344812",
            "extra": "mean: 153.83395399999245 msec\nrounds: 6"
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
          "id": "b22ad3dc514de7c5fbfc7878b9a33b50e5215f77",
          "message": "Fix semeio test in CI\n\nAvoids pulling semeio from pypi after having installed it\nfrom source",
          "timestamp": "2024-02-20T07:25:48+01:00",
          "tree_id": "59ed468833667c7796af349f17e8d3f07b893bc8",
          "url": "https://github.com/equinor/ert/commit/b22ad3dc514de7c5fbfc7878b9a33b50e5215f77"
        },
        "date": 1708410526220,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.703428274773947,
            "unit": "iter/sec",
            "range": "stddev: 0.032553266851892255",
            "extra": "mean: 149.1773998333296 msec\nrounds: 6"
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
          "id": "f98778c16a406a4fec8f6ee841f03d8606d367ff",
          "message": "Update general observation error message when no timemap or reference is provided.",
          "timestamp": "2024-02-20T10:57:25+02:00",
          "tree_id": "7dd3aab8fedfbca7eb4f8a7575e3ffea93199b9b",
          "url": "https://github.com/equinor/ert/commit/f98778c16a406a4fec8f6ee841f03d8606d367ff"
        },
        "date": 1708419598889,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.776365850173066,
            "unit": "iter/sec",
            "range": "stddev: 0.0320220659989589",
            "extra": "mean: 147.5717253333452 msec\nrounds: 6"
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
          "id": "d35aa943f8940ebe706796827c789092e69ab17e",
          "message": "Document choice of rule names and bundles",
          "timestamp": "2024-02-20T15:06:22+01:00",
          "tree_id": "cdd866ec4ca469243b459f55a00af76fdede9728",
          "url": "https://github.com/equinor/ert/commit/d35aa943f8940ebe706796827c789092e69ab17e"
        },
        "date": 1708438154239,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.729504265146314,
            "unit": "iter/sec",
            "range": "stddev: 0.03038872109678893",
            "extra": "mean: 148.59935600000065 msec\nrounds: 6"
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
          "id": "16a5bb40b57d8e7c8a5f03e791ed71bd281b011e",
          "message": "Set default pytest timeout of 240s (#7218)\n\nSet default pytest timeout to 360s",
          "timestamp": "2024-02-22T07:26:27+01:00",
          "tree_id": "3e2e4b0e391b2faff665ded2a3fdc6cfb09d098b",
          "url": "https://github.com/equinor/ert/commit/16a5bb40b57d8e7c8a5f03e791ed71bd281b011e"
        },
        "date": 1708583372664,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.690445871685375,
            "unit": "iter/sec",
            "range": "stddev: 0.033404854765318924",
            "extra": "mean: 149.46686949999824 msec\nrounds: 6"
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
          "id": "ff0d6c9d4f6857603497557936f839e7c45f90f0",
          "message": "Change log from jobqueue to match scheduler",
          "timestamp": "2024-02-22T08:29:12+01:00",
          "tree_id": "dd5a9b518ca5b6afcb867635636201dd5e51d7b2",
          "url": "https://github.com/equinor/ert/commit/ff0d6c9d4f6857603497557936f839e7c45f90f0"
        },
        "date": 1708587135702,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.527316665766304,
            "unit": "iter/sec",
            "range": "stddev: 0.03729702481941187",
            "extra": "mean: 153.2023113333357 msec\nrounds: 6"
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
          "id": "28cce9f6d4222c21753424770208b4b93aff02cc",
          "message": "Add missing space in ConfigWarning msg",
          "timestamp": "2024-02-22T08:39:31+01:00",
          "tree_id": "3f82b954e834e041a13cb80c2ef8f8a1bc64d417",
          "url": "https://github.com/equinor/ert/commit/28cce9f6d4222c21753424770208b4b93aff02cc"
        },
        "date": 1708587720074,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.736012281492522,
            "unit": "iter/sec",
            "range": "stddev: 0.03253703465560063",
            "extra": "mean: 148.4557863333388 msec\nrounds: 6"
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
          "id": "a77ee9ca669cce6c75c6ddd448997ee313b9f677",
          "message": "Ensure Summary Response is not empty",
          "timestamp": "2024-02-22T11:35:07+01:00",
          "tree_id": "c12dad92c2efee170d7c579fdc348d478cde18cf",
          "url": "https://github.com/equinor/ert/commit/a77ee9ca669cce6c75c6ddd448997ee313b9f677"
        },
        "date": 1708598267852,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.448519617704249,
            "unit": "iter/sec",
            "range": "stddev: 0.03932796380538554",
            "extra": "mean: 155.0743518333301 msec\nrounds: 6"
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
          "id": "92023ea36b9a059ecb6c5ca0d2e9ddc6c83254fb",
          "message": "Clarify error for non-UTF-8 encoded data in runpath setup\n\n* Update docs of DATA_FILE\r\n\r\nPreviously, if a data file's encoding was changed to non-UTF-8\r\nafter initial configuration parsing, it would lead to cryptic\r\ncrashes upon subsequent runpath creation.",
          "timestamp": "2024-02-22T12:52:21+01:00",
          "tree_id": "2fc5894c662ead25725915c71ed15794aa6ea5aa",
          "url": "https://github.com/equinor/ert/commit/92023ea36b9a059ecb6c5ca0d2e9ddc6c83254fb"
        },
        "date": 1708602894392,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.352366365784346,
            "unit": "iter/sec",
            "range": "stddev: 0.03691438618385361",
            "extra": "mean: 157.4216508333469 msec\nrounds: 6"
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
          "id": "fcad7df2f8778f6db2c5b68c9c352a1670f0bf73",
          "message": "Reduce logging level of most noisy logs",
          "timestamp": "2024-02-22T14:01:58+01:00",
          "tree_id": "f571f7af0fcf6e2d656ecb1e9d29c2f1551e4974",
          "url": "https://github.com/equinor/ert/commit/fcad7df2f8778f6db2c5b68c9c352a1670f0bf73"
        },
        "date": 1708607156737,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.681623361778011,
            "unit": "iter/sec",
            "range": "stddev: 0.03262894160939757",
            "extra": "mean: 149.66422766665724 msec\nrounds: 6"
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
          "id": "1ff816db9ef94737317eabc77ced6d7daff9a02d",
          "message": "Support (in new PBS driver) and deprecate JOB_PREFIX\n\nJOB_PREFIX is not mentioned in the docs and no usage is known\nin the logs. It should also be redundant.",
          "timestamp": "2024-02-22T14:05:52+01:00",
          "tree_id": "9a68cbd8894f69a40c4ebe7f3492a2b194753243",
          "url": "https://github.com/equinor/ert/commit/1ff816db9ef94737317eabc77ced6d7daff9a02d"
        },
        "date": 1708607319373,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6285952634847085,
            "unit": "iter/sec",
            "range": "stddev: 0.03524201960578196",
            "extra": "mean: 150.8615265000041 msec\nrounds: 6"
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
      }
    ]
  }
}