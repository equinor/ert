window.BENCHMARK_DATA = {
  "lastUpdate": 1709130826613,
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
          "id": "4807e3177df8340d36e5838cde71d872fb08f29b",
          "message": "Storage Server no longer exists so delete it from docs",
          "timestamp": "2024-02-27T14:25:47+01:00",
          "tree_id": "f1d1a330f51b9d02dfc7f57e2010459b68faa259",
          "url": "https://github.com/equinor/ert/commit/4807e3177df8340d36e5838cde71d872fb08f29b"
        },
        "date": 1709040517781,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4776695520716587,
            "unit": "iter/sec",
            "range": "stddev: 0.5043383750159872",
            "extra": "mean: 2.093497472600018 sec\nrounds: 5"
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
          "id": "8a755ca13cde9e9cedabc703fda5a4b6eb5240e4",
          "message": "Fix the codecov retry\n\nWas missing contine-on-error",
          "timestamp": "2024-02-27T15:08:35+01:00",
          "tree_id": "db59e84392616d3b6989903e095ca2adfd7bb6d1",
          "url": "https://github.com/equinor/ert/commit/8a755ca13cde9e9cedabc703fda5a4b6eb5240e4"
        },
        "date": 1709043074899,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4448675296385362,
            "unit": "iter/sec",
            "range": "stddev: 0.5432989339982475",
            "extra": "mean: 2.2478601682000035 sec\nrounds: 5"
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
          "id": "571e94fcf1c0ee971cda1c64c56b282e6ea91c37",
          "message": "Use lowercase for observation dataclasses",
          "timestamp": "2024-02-27T15:34:16+01:00",
          "tree_id": "0f62e2abf1f3a16c9f68fa967960ab907a90b430",
          "url": "https://github.com/equinor/ert/commit/571e94fcf1c0ee971cda1c64c56b282e6ea91c37"
        },
        "date": 1709044630590,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4586832132533414,
            "unit": "iter/sec",
            "range": "stddev: 0.5260028956872186",
            "extra": "mean: 2.180153908199986 sec\nrounds: 5"
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
          "id": "759f431ec6827a149663ece6691672c8b5122057",
          "message": "Remove unused state TIME_MAP_FAILURE",
          "timestamp": "2024-02-27T16:38:49+01:00",
          "tree_id": "ce6b53fbaed38d406f64393fa20d9aedae0b6af7",
          "url": "https://github.com/equinor/ert/commit/759f431ec6827a149663ece6691672c8b5122057"
        },
        "date": 1709048521308,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4596911028553359,
            "unit": "iter/sec",
            "range": "stddev: 0.39809382995724285",
            "extra": "mean: 2.1753738408000003 sec\nrounds: 5"
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
          "id": "5faa55b8a32e87844aea59f3858227ba8a467e46",
          "message": "Forward exceptions from threads to main thread\n\nWe substitute usages of `threading.Thread` with `ErtThread`. These are\nmostly equivalent, with the exception that `ErtThread` catches all\nexceptions that are propagated to the `run` function. This exception is\nthen saved globally before we raise a `SIGUSR1` signal to ourselves so\nthat the main thread can wake up and raise the signal from its context.\nIn practice, this means that any part of Ert can throw an exception and\nhave it be propagated to `cli.main` to be processed there.\n\nIt should also be noted that returning from `SIGUSR1` handler is\npermitted, and so eg. Ert GUI could continue rather than quit.",
          "timestamp": "2024-02-27T17:15:43+01:00",
          "tree_id": "be2b0be4a778e8adb72c9cc38f90db2846ab94f3",
          "url": "https://github.com/equinor/ert/commit/5faa55b8a32e87844aea59f3858227ba8a467e46"
        },
        "date": 1709050709000,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4334661250334278,
            "unit": "iter/sec",
            "range": "stddev: 0.4655133737825086",
            "extra": "mean: 2.3069853496000006 sec\nrounds: 5"
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
          "id": "20bb4660976814340e22d2b03162b4f58191aca4",
          "message": "Fix various warnings from unit tests",
          "timestamp": "2024-02-28T09:52:23+01:00",
          "tree_id": "6da92395e5ac1f6332dab330fa01ecca7083f949",
          "url": "https://github.com/equinor/ert/commit/20bb4660976814340e22d2b03162b4f58191aca4"
        },
        "date": 1709110505507,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.44333550117891185,
            "unit": "iter/sec",
            "range": "stddev: 0.4866452798935534",
            "extra": "mean: 2.2556280679999987 sec\nrounds: 5"
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
          "id": "a820df03b182253f78c917697cc277c07b6e1b7f",
          "message": "Update reference to ert-testdata submodule",
          "timestamp": "2024-02-28T12:48:04+01:00",
          "tree_id": "8cf0398d0d831d3dc429f956fbbb694a581ee646",
          "url": "https://github.com/equinor/ert/commit/a820df03b182253f78c917697cc277c07b6e1b7f"
        },
        "date": 1709121071450,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.45232891317585905,
            "unit": "iter/sec",
            "range": "stddev: 0.41502633749408724",
            "extra": "mean: 2.2107806307999907 sec\nrounds: 5"
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
          "id": "ebe74e4a79d3e5ed208ac83212ac33546ce83d3a",
          "message": "Add support for CLUSTER_LABEL in PBS",
          "timestamp": "2024-02-28T12:59:09+01:00",
          "tree_id": "8df147630a7c8b22506a143a390273e3f351dab6",
          "url": "https://github.com/equinor/ert/commit/ebe74e4a79d3e5ed208ac83212ac33546ce83d3a"
        },
        "date": 1709121737479,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.44755395600082143,
            "unit": "iter/sec",
            "range": "stddev: 0.48180133682557125",
            "extra": "mean: 2.2343674691999924 sec\nrounds: 5"
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
          "id": "96fd75918d0ea260f17e95e6c2b22e646ff5ee02",
          "message": "Ensure NUM_CPU is accompanied by queue config settings for Torque\n\nIf the users chooses to set NUM_CPU to something larger than 1\nand is using the Torque, the user must specify how those CPUs\nare to be distributed over nodes.",
          "timestamp": "2024-02-28T14:08:28+01:00",
          "tree_id": "b87963ad93273a099d78b1aa8ac7d72155ddabe1",
          "url": "https://github.com/equinor/ert/commit/96fd75918d0ea260f17e95e6c2b22e646ff5ee02"
        },
        "date": 1709125898386,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4289187360449633,
            "unit": "iter/sec",
            "range": "stddev: 0.4605135875618919",
            "extra": "mean: 2.3314439682000057 sec\nrounds: 5"
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
          "id": "3307b7ca5eebc6eaaf09f746192347cb2a564f35",
          "message": "Fix multiple done_callbacks when retrying\n\nThe scheduler code ran the done callback for every attempt at running a realization.\nWe only want to try internalization when a realization finishes with a zero return code,\nso when we need MAX_SUBMIT to obtain zero returncode, the done callback is\nonly executed after the final attempt.",
          "timestamp": "2024-02-28T14:11:45+01:00",
          "tree_id": "833b92e5dc9121bfe59a5404b446657049fb6977",
          "url": "https://github.com/equinor/ert/commit/3307b7ca5eebc6eaaf09f746192347cb2a564f35"
        },
        "date": 1709126076976,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4392261434036039,
            "unit": "iter/sec",
            "range": "stddev: 0.47790150120376723",
            "extra": "mean: 2.276731508399996 sec\nrounds: 5"
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
          "id": "c519fea65938878aa402886fa27caea6468dd9b5",
          "message": "Use a newer block storage containing fixes for some deprecation warnings",
          "timestamp": "2024-02-28T15:22:21+01:00",
          "tree_id": "6b8487bc3aff69c92541ba05e6d77f9f70b07c27",
          "url": "https://github.com/equinor/ert/commit/c519fea65938878aa402886fa27caea6468dd9b5"
        },
        "date": 1709130305709,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.42704979024759016,
            "unit": "iter/sec",
            "range": "stddev: 0.5522304717452802",
            "extra": "mean: 2.3416473273999996 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b3323c371df0b8f23851c97d59148b59374ba6ad",
          "message": "Add tests for LSF driver kills (#7202)\n\nThe driver will never retry killing but if it does not succeed\r\nthe error messages are logged.\r\n\r\nKilling twice will give a logged error if the first is successful.",
          "timestamp": "2024-02-28T15:30:55+01:00",
          "tree_id": "e3b78008b42bf1b460af5393951b059e367b7992",
          "url": "https://github.com/equinor/ert/commit/b3323c371df0b8f23851c97d59148b59374ba6ad"
        },
        "date": 1709130826119,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4516404335599372,
            "unit": "iter/sec",
            "range": "stddev: 0.49800077308195023",
            "extra": "mean: 2.214150739600001 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}