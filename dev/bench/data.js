window.BENCHMARK_DATA = {
  "lastUpdate": 1707748758934,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "b35134a8693e851ee67ca8bc7d8ffefeb413c3e0",
          "message": "Pin hypothesis version",
          "timestamp": "2024-02-05T10:01:04+01:00",
          "tree_id": "bbbdd4babff83ed4398cb318f62c44dabc40ad85",
          "url": "https://github.com/equinor/ert/commit/b35134a8693e851ee67ca8bc7d8ffefeb413c3e0"
        },
        "date": 1707123830297,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 7.063719401639363,
            "unit": "iter/sec",
            "range": "stddev: 0.0033657036562662708",
            "extra": "mean: 141.56847733333203 msec\nrounds: 6"
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
          "id": "faf029d80889a2c2a763f9cb16d749f885800c86",
          "message": "Fix test data generation failing\n\nhypothesis sometimes fails a health check with\ntoo much test data removed due to failed assumption.\n\nThe failing assumption is changed to an assignment",
          "timestamp": "2024-02-05T10:05:55+01:00",
          "tree_id": "35f7bb0dc7f7c0c04d4a83b14475366ffa22cc0b",
          "url": "https://github.com/equinor/ert/commit/faf029d80889a2c2a763f9cb16d749f885800c86"
        },
        "date": 1707124159461,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 7.089229752918906,
            "unit": "iter/sec",
            "range": "stddev: 0.004654904784152193",
            "extra": "mean: 141.05904799999772 msec\nrounds: 6"
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
          "id": "a1eb1b922b2d1c84cba9271282b783dc628ae7dd",
          "message": "Unpin hypothesis version",
          "timestamp": "2024-02-05T10:42:17+01:00",
          "tree_id": "778fa9459c0358f1ab0a1af35d1085d6e25e984c",
          "url": "https://github.com/equinor/ert/commit/a1eb1b922b2d1c84cba9271282b783dc628ae7dd"
        },
        "date": 1707126294145,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.443329352082363,
            "unit": "iter/sec",
            "range": "stddev: 0.032325512951918545",
            "extra": "mean: 155.19926816666896 msec\nrounds: 6"
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
          "id": "e35e7e07b39534ac3ed83085a87a970b633dcc70",
          "message": "Update codecov uploader",
          "timestamp": "2024-02-05T11:01:42+01:00",
          "tree_id": "a8e76883b5e07552854c458080376f601a0bb5bd",
          "url": "https://github.com/equinor/ert/commit/e35e7e07b39534ac3ed83085a87a970b633dcc70"
        },
        "date": 1707127469304,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.690313404812423,
            "unit": "iter/sec",
            "range": "stddev: 0.08454328250779851",
            "extra": "mean: 175.7372448333475 msec\nrounds: 6"
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
          "id": "db56a48697a130591edb4688ca52992d589972d9",
          "message": "Replace some tests with an integration test\n\nAvoids relying on details. Testing of storage api is\ndelegated to the StorageTest RuleBasedStateMachine.",
          "timestamp": "2024-02-05T13:47:20+01:00",
          "tree_id": "f735293ba9f5f6b8278699b574293cb2d4159e50",
          "url": "https://github.com/equinor/ert/commit/db56a48697a130591edb4688ca52992d589972d9"
        },
        "date": 1707137436454,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.467002087358255,
            "unit": "iter/sec",
            "range": "stddev: 0.03205177609819873",
            "extra": "mean: 154.63115466667432 msec\nrounds: 6"
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
          "id": "87e3593146d4c04be34eadad3a6cd09760a20b64",
          "message": "Remove setAlignment that just loops forever",
          "timestamp": "2024-02-05T14:40:50+01:00",
          "tree_id": "49379e1ccdec54a55ce0257fb1dabeaed5d4e783",
          "url": "https://github.com/equinor/ert/commit/87e3593146d4c04be34eadad3a6cd09760a20b64"
        },
        "date": 1707140612433,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.388497311270369,
            "unit": "iter/sec",
            "range": "stddev: 0.03314286899210945",
            "extra": "mean: 156.53133299998956 msec\nrounds: 6"
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
          "id": "660b39f22daf94299391d1cc56580caff6d85f15",
          "message": "Remove unused ERT_HAS_REGEXP",
          "timestamp": "2024-02-05T14:41:12+01:00",
          "tree_id": "cb5f5906883d81881206f376190dd5a051917592",
          "url": "https://github.com/equinor/ert/commit/660b39f22daf94299391d1cc56580caff6d85f15"
        },
        "date": 1707140620118,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.378298999355107,
            "unit": "iter/sec",
            "range": "stddev: 0.03275143515782937",
            "extra": "mean: 156.78161216667755 msec\nrounds: 6"
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
          "id": "acd8bfe9f1ffb86e3113dbef969d9b308d975bec",
          "message": "Fix ert dark storage performance tests (#7109)\n\nFixed ert dark storage performance tests timing out by changing src/ert/dark_storage/common.py get_observations_for_obs_keys(). Moved observations dict outside of loop to only generate dict once instead of every loop iteration.",
          "timestamp": "2024-02-05T14:18:47Z",
          "tree_id": "ee890b9d54b8d55dfa4c8a6db35e865d09873b51",
          "url": "https://github.com/equinor/ert/commit/acd8bfe9f1ffb86e3113dbef969d9b308d975bec"
        },
        "date": 1707142892351,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.23721244106503,
            "unit": "iter/sec",
            "range": "stddev: 0.03970188347404076",
            "extra": "mean: 160.32803266666443 msec\nrounds: 6"
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
          "id": "c91ddc8770420422256583bca8bc4ec21b29f445",
          "message": "pin iterative_ensemble_smoother",
          "timestamp": "2024-02-06T15:30:45+01:00",
          "tree_id": "3bed58d12ae40d7d316ffb030fd8c17547a85337",
          "url": "https://github.com/equinor/ert/commit/c91ddc8770420422256583bca8bc4ec21b29f445"
        },
        "date": 1707230007826,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.445836421864466,
            "unit": "iter/sec",
            "range": "stddev: 0.030921530725637712",
            "extra": "mean: 155.13890433334154 msec\nrounds: 6"
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
          "id": "9c2b60099a54eeb5bb40013acef721e30558a86c",
          "message": "Fix a performance problem with summary reading",
          "timestamp": "2024-02-06T19:26:10+01:00",
          "tree_id": "65e89a58aee29747e6ad03a1bf4359771d0fc2ed",
          "url": "https://github.com/equinor/ert/commit/9c2b60099a54eeb5bb40013acef721e30558a86c"
        },
        "date": 1707244126770,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.379713687742706,
            "unit": "iter/sec",
            "range": "stddev: 0.034842893034089065",
            "extra": "mean: 156.74684616666923 msec\nrounds: 6"
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
          "id": "7522112859895b7717096dafa22943445eb56168",
          "message": "Get progress from adaptive localization",
          "timestamp": "2024-02-07T06:46:30+01:00",
          "tree_id": "ca6339f75dacb56fb243e8d5170ee8f1e2f0dac9",
          "url": "https://github.com/equinor/ert/commit/7522112859895b7717096dafa22943445eb56168"
        },
        "date": 1707284943440,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.154777709437055,
            "unit": "iter/sec",
            "range": "stddev: 0.03621495910803309",
            "extra": "mean: 162.47540483333958 msec\nrounds: 6"
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
          "id": "63dfa327980e31af0b1ea9f75198bca425bb5fb9",
          "message": "Improve plotting performance in GUI",
          "timestamp": "2024-02-07T08:00:13+01:00",
          "tree_id": "daf980a485b3902dec9870dabd7826715cc4e305",
          "url": "https://github.com/equinor/ert/commit/63dfa327980e31af0b1ea9f75198bca425bb5fb9"
        },
        "date": 1707289373988,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.34948817241497,
            "unit": "iter/sec",
            "range": "stddev: 0.0342605196076938",
            "extra": "mean: 157.49300933332697 msec\nrounds: 6"
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
          "id": "5e1a4223a1e1f598535666e7fcc66c14af769aeb",
          "message": "Fix slow observation parsing\n\nNote that this potentially means that some\nambiguties regarding \"--\" in values are\nno longer considered valid.",
          "timestamp": "2024-02-07T09:13:52+01:00",
          "tree_id": "5a72e3dd1f2f53eeaccaeab2c99c105072886281",
          "url": "https://github.com/equinor/ert/commit/5e1a4223a1e1f598535666e7fcc66c14af769aeb"
        },
        "date": 1707293785298,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.204547007444889,
            "unit": "iter/sec",
            "range": "stddev: 0.04402732098241109",
            "extra": "mean: 161.17212083333263 msec\nrounds: 6"
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
          "id": "79364799602ddccb954a9c2e353bb27f22c54a0c",
          "message": "Remove benchmarking files not in use (#7111)\n\nRemove benchmarking files created by dark storage performance benchmarks. They are not in use as we are not comparing results anymore.",
          "timestamp": "2024-02-07T10:36:36Z",
          "tree_id": "7c9781038668eccea9411e8b049b5cf280438833",
          "url": "https://github.com/equinor/ert/commit/79364799602ddccb954a9c2e353bb27f22c54a0c"
        },
        "date": 1707302360897,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.311797666943451,
            "unit": "iter/sec",
            "range": "stddev: 0.03566877695823342",
            "extra": "mean: 158.43346900000674 msec\nrounds: 6"
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
          "id": "c1815d31fb1e8b0c8712762e500be50b11d19aa7",
          "message": "Refactor to Native Python BlockFs migration",
          "timestamp": "2024-02-08T12:07:00+01:00",
          "tree_id": "d7070fb630472ada63469380de3a5e619616f404",
          "url": "https://github.com/equinor/ert/commit/c1815d31fb1e8b0c8712762e500be50b11d19aa7"
        },
        "date": 1707390575671,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.990836036948234,
            "unit": "iter/sec",
            "range": "stddev: 0.0030636012610722727",
            "extra": "mean: 143.04440766665985 msec\nrounds: 6"
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
          "id": "dddf883a9c4fb7d53385c11a34a8d5ede1c63ee5",
          "message": "Fix bug where one missing index.json caused no ensembles to be loaded\n\nThis is not without pitfalls, as the missing ensemble could be related to\nan ensemble that still exists.",
          "timestamp": "2024-02-08T12:29:55+01:00",
          "tree_id": "36b827e00f718f95fce5c974b947431e1665a068",
          "url": "https://github.com/equinor/ert/commit/dddf883a9c4fb7d53385c11a34a8d5ede1c63ee5"
        },
        "date": 1707391974978,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.815389111811542,
            "unit": "iter/sec",
            "range": "stddev: 0.004678414036537113",
            "extra": "mean: 146.72676549999628 msec\nrounds: 6"
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
          "id": "7146cf91053db43444c894ea0690a9d90eb2d1ab",
          "message": "Hardcode text color to black in suggestor window",
          "timestamp": "2024-02-08T13:54:12+01:00",
          "tree_id": "a5b346917406b8ce8377b150aa7560b8081b8b23",
          "url": "https://github.com/equinor/ert/commit/7146cf91053db43444c894ea0690a9d90eb2d1ab"
        },
        "date": 1707397011718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.381333968517861,
            "unit": "iter/sec",
            "range": "stddev: 0.029039350187174195",
            "extra": "mean: 156.70704666664884 msec\nrounds: 6"
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
          "id": "de7c7ed7e4298e086a9775e1734275a3e175eadd",
          "message": "Filter out deprecation warnings from main entry point",
          "timestamp": "2024-02-08T14:57:48+01:00",
          "tree_id": "41e0e926c8429ca61569ef2df7b9e0bc66b94b16",
          "url": "https://github.com/equinor/ert/commit/de7c7ed7e4298e086a9775e1734275a3e175eadd"
        },
        "date": 1707400822879,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 7.03756056681585,
            "unit": "iter/sec",
            "range": "stddev: 0.004347485605542682",
            "extra": "mean: 142.09469183331672 msec\nrounds: 6"
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
          "id": "a9449390663748cc5e47aa6002cfbe2e9fdc9f4f",
          "message": "Add deprecation for INVERSION with number",
          "timestamp": "2024-02-08T15:50:04+01:00",
          "tree_id": "ba3bf4f952c3e8ea04cd6e7259e0308693bfc5ef",
          "url": "https://github.com/equinor/ert/commit/a9449390663748cc5e47aa6002cfbe2e9fdc9f4f"
        },
        "date": 1707403967055,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.842551807488977,
            "unit": "iter/sec",
            "range": "stddev: 0.01125754937724104",
            "extra": "mean: 146.14430816665921 msec\nrounds: 6"
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
          "id": "d30e7c9863f0c1ae6323db64b8f3e19dac7f389b",
          "message": "Replace Reader/Accessor with a Mode enum\n\nThis commit extends the \"mode\" concept and replaces the Reader/Accessor\npattern that was in `ert.storage` previously. This is to make it\npossible to delete ensembles and experiments in a safe way.\n\nThe modes are `READ` (`\"r\"`) and `WRITE` (`\"w\"`). The -Reader classes\nare equivalent to `READ` and the -Accessor classes are equivalent to\n`WRITE`. With this change it is possible to downgrade the access-level\nof objects without having to reopen them. This is in particular useful\nwhen deleting ensembles and experiments.\n\nSuppose there is a new method called `.delete_ensemble` which does what\nit says. Consider the following code using the Reader/Accessor pattern:\n\n```py\nwith open_storage(path, mode=\"w\") as storage:\n    ens = storage.get_ensemble_by_name(\"default\")\n    storage.delete_ensemble(ens)\n\n    # ... sometime later:\n\n    # ens is EnsembleAccessor, so writing is valid:\n    ens.save_responses(group, real, data)\n```\n\nUh-oh! We've accidentally saved data to a deleted object. Now, consider\nwith the capability pattern:\n\n```py\nwith open_storage(path, mode=\"w\") as storage:\n    ens = storage.get_ensemble_by_name(\"default\")\n    storage.delete_ensemble(ens)\n\n    # ... sometime later:\n\n    # ens was reduced to Mode.NONE. Calling this method raises ModeError.\n    ens.save_responses(group, real, data)\n```\n\nThis should also make it easier to understand the module.",
          "timestamp": "2024-02-08T19:49:54+01:00",
          "tree_id": "419a891d7e9b59feaf981500176854f9ede2f710",
          "url": "https://github.com/equinor/ert/commit/d30e7c9863f0c1ae6323db64b8f3e19dac7f389b"
        },
        "date": 1707418346260,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.980348586439082,
            "unit": "iter/sec",
            "range": "stddev: 0.005714571810297668",
            "extra": "mean: 143.2593211666718 msec\nrounds: 6"
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
          "id": "1bc352403170dc73c6d7314bd921e5ae30371add",
          "message": "Update observations docs",
          "timestamp": "2024-02-09T06:34:38+01:00",
          "tree_id": "68a0acd57993f207ea6bd9b22d65292d61c018c1",
          "url": "https://github.com/equinor/ert/commit/1bc352403170dc73c6d7314bd921e5ae30371add"
        },
        "date": 1707457048474,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 7.029310250853047,
            "unit": "iter/sec",
            "range": "stddev: 0.003912285437337956",
            "extra": "mean: 142.2614686666653 msec\nrounds: 6"
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
          "id": "5072a4eb4c17a20d45efa81df191c5728518d5fe",
          "message": "Remove remove button from Manage Cases\n\nWe want to support this but needs design",
          "timestamp": "2024-02-09T09:15:21+01:00",
          "tree_id": "587528d5a5615b9837a5fef340afa7f50674d42a",
          "url": "https://github.com/equinor/ert/commit/5072a4eb4c17a20d45efa81df191c5728518d5fe"
        },
        "date": 1707466674129,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.787956828264091,
            "unit": "iter/sec",
            "range": "stddev: 0.008626787422084824",
            "extra": "mean: 147.31973483333624 msec\nrounds: 6"
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
          "id": "791fe11db656f345b4829c91a2e08ca6a2f58e1e",
          "message": "Fix a regression of using REFCASE with extension",
          "timestamp": "2024-02-09T13:13:24+01:00",
          "tree_id": "fd2571a5704d129ca897ac5cf522131c50a68b2c",
          "url": "https://github.com/equinor/ert/commit/791fe11db656f345b4829c91a2e08ca6a2f58e1e"
        },
        "date": 1707480962848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.964415955081603,
            "unit": "iter/sec",
            "range": "stddev: 0.0039030350123011497",
            "extra": "mean: 143.58705833334776 msec\nrounds: 6"
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
          "id": "9cade6d768112657f7824643b63b999bcb0db271",
          "message": "Add backup path in torque mock binaries (#7152)\n\nBefore this commit, the torque mock binaries put the jobs in the root directory if PYTEST_TMP_PATH was not set. This commit sets the cwd as backup instead.",
          "timestamp": "2024-02-09T15:12:01+01:00",
          "tree_id": "e959821963cc81690e8bf21c620c48d15f23144f",
          "url": "https://github.com/equinor/ert/commit/9cade6d768112657f7824643b63b999bcb0db271"
        },
        "date": 1707488076163,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.95231707449157,
            "unit": "iter/sec",
            "range": "stddev: 0.004510065809165687",
            "extra": "mean: 143.8369380000021 msec\nrounds: 6"
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
          "id": "ea72fd7da2b03720e5a49e2a5eecf0f3b0bfa36d",
          "message": "Store experiment name in index.json",
          "timestamp": "2024-02-12T09:29:25+01:00",
          "tree_id": "5cb098bec29499ae524f6da48a38bbf07111825f",
          "url": "https://github.com/equinor/ert/commit/ea72fd7da2b03720e5a49e2a5eecf0f3b0bfa36d"
        },
        "date": 1707726740206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.392287253707158,
            "unit": "iter/sec",
            "range": "stddev: 0.03339827768272757",
            "extra": "mean: 156.43852666665717 msec\nrounds: 6"
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
          "id": "a4a5cc71ff58d900471a172dd9a3a4b55763198e",
          "message": "Ensure local queue in tests with site config\n\nsite config can be set by environment, so ensure\ntests run local queue even though site config\nspecifies e.g. LSF.",
          "timestamp": "2024-02-12T13:09:39+01:00",
          "tree_id": "894056e1d6a89bd166a3916b3413d2779b85361b",
          "url": "https://github.com/equinor/ert/commit/a4a5cc71ff58d900471a172dd9a3a4b55763198e"
        },
        "date": 1707739948178,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.265737744868947,
            "unit": "iter/sec",
            "range": "stddev: 0.038707241546828565",
            "extra": "mean: 159.59812566667134 msec\nrounds: 6"
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
          "id": "77411f5b25308932019bc0d832994bc957019798",
          "message": "Always say job has started if an event is received\n\nIf the queue system is very quick and the polling time period is long\nenough, the driver will never observe the job while it is in its\nrunning state.",
          "timestamp": "2024-02-12T14:16:13+01:00",
          "tree_id": "7e7de11d061b02a0fae1d58867f6379e47e0813d",
          "url": "https://github.com/equinor/ert/commit/77411f5b25308932019bc0d832994bc957019798"
        },
        "date": 1707743947417,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.459292766178555,
            "unit": "iter/sec",
            "range": "stddev: 0.032013837526842934",
            "extra": "mean: 154.81571066666788 msec\nrounds: 6"
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
          "id": "34fa32c648028f260dfebb26be09a1a9433f34c8",
          "message": "Fix incorrect note statement",
          "timestamp": "2024-02-12T14:37:44+01:00",
          "tree_id": "e4571e585f6a505a96773259f498e54afdc51000",
          "url": "https://github.com/equinor/ert/commit/34fa32c648028f260dfebb26be09a1a9433f34c8"
        },
        "date": 1707745212768,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.41527770843309,
            "unit": "iter/sec",
            "range": "stddev: 0.034776247601240456",
            "extra": "mean: 155.87789733334034 msec\nrounds: 6"
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
          "id": "a1a2bc34ff0bf878b72a6b27852771f42cab07ed",
          "message": "Add logging statements",
          "timestamp": "2024-02-12T14:49:33+01:00",
          "tree_id": "d26db43a30a4439b042c8fd934a4493ca5cd66ea",
          "url": "https://github.com/equinor/ert/commit/a1a2bc34ff0bf878b72a6b27852771f42cab07ed"
        },
        "date": 1707745931298,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.6068730582495245,
            "unit": "iter/sec",
            "range": "stddev: 0.04978685764989781",
            "extra": "mean: 178.35253083332722 msec\nrounds: 6"
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
          "id": "942555d4ad6834c2b66741354fc73f18c7f94e11",
          "message": "Fix bug causing FinishedEvent to be ignored\n\nOnly applies to new scheduler and non-local drivers",
          "timestamp": "2024-02-12T15:36:45+01:00",
          "tree_id": "f131b463894b58719b2a3710cd092065515a14ea",
          "url": "https://github.com/equinor/ert/commit/942555d4ad6834c2b66741354fc73f18c7f94e11"
        },
        "date": 1707748758366,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.567501739545185,
            "unit": "iter/sec",
            "range": "stddev: 0.028483380302981167",
            "extra": "mean: 152.26490066666543 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}