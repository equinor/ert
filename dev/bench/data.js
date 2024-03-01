window.BENCHMARK_DATA = {
  "lastUpdate": 1709278762620,
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
          "id": "4b81076a518a4d7998ac50a98acab7227f0c6bd4",
          "message": "Add values to analysis_module",
          "timestamp": "2024-02-29T12:07:28+01:00",
          "tree_id": "dcf718e0abe55f9bb02a78e01267fed02bfa9278",
          "url": "https://github.com/equinor/ert/commit/4b81076a518a4d7998ac50a98acab7227f0c6bd4"
        },
        "date": 1709205024762,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4484463537954822,
            "unit": "iter/sec",
            "range": "stddev: 0.43739118847639785",
            "extra": "mean: 2.2299211299999966 sec\nrounds: 5"
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
          "id": "ad79d0e111b438c6c925813e0161e4a7073e7d9c",
          "message": "Add a test for temporary storage of Fields",
          "timestamp": "2024-02-29T12:19:32+01:00",
          "tree_id": "bf66901699ba71319cd4d8016368bc6f855a939b",
          "url": "https://github.com/equinor/ert/commit/ad79d0e111b438c6c925813e0161e4a7073e7d9c"
        },
        "date": 1709206065593,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.37697947290027084,
            "unit": "iter/sec",
            "range": "stddev: 0.6024723436954574",
            "extra": "mean: 2.6526643276000015 sec\nrounds: 5"
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
          "id": "67c65f36ac17cadba09c05846cdb10308a09ec79",
          "message": "Update ruff config in pyproject.toml (#7297)\n\nFollowing up recommendation from ruff itself:\r\nwarning: The top-level linter settings are deprecated in favour of their counterparts in the  section. Please update the following options in :\r\n  - 'ignore' -> 'lint.ignore'\r\n  - 'select' -> 'lint.select'\r\n  - 'pylint' -> 'lint.pylint'\r\n  - 'extend-per-file-ignores' -> 'lint.extend-per-file-ignores'",
          "timestamp": "2024-02-29T12:58:14+01:00",
          "tree_id": "a09682844f414b7393ee5b21b8cd355e5703065f",
          "url": "https://github.com/equinor/ert/commit/67c65f36ac17cadba09c05846cdb10308a09ec79"
        },
        "date": 1709208064876,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.42933555584756994,
            "unit": "iter/sec",
            "range": "stddev: 0.4618848778768037",
            "extra": "mean: 2.329180489200007 sec\nrounds: 5"
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
          "id": "c6ce6839bbb0bd36d0cfc77554f5159bd63365b6",
          "message": "Use codecs.decode() with errors ignored (#7298)\n\nThis avoids needing to handle UnicodeDecodeError exceptions",
          "timestamp": "2024-02-29T12:58:29+01:00",
          "tree_id": "73e75bcc05dfc5720e5d6786be582dbfca7a07e2",
          "url": "https://github.com/equinor/ert/commit/c6ce6839bbb0bd36d0cfc77554f5159bd63365b6"
        },
        "date": 1709208107383,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4238393979995127,
            "unit": "iter/sec",
            "range": "stddev: 0.4588346274191951",
            "extra": "mean: 2.3593842496000095 sec\nrounds: 5"
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
          "id": "3575519b1da29b656682fde441389deb608cbe78",
          "message": "Add tests for various scenarios for a faulty bjobs (#7193)\n\nAdd tests for faulty bjobs behaviour\r\n\r\nCurrently no result from bjobs will take down Ert.\r\n\r\nAlso, there is currently no retry functionality for some valid\r\nerror scenarios, this is to be fixed later.",
          "timestamp": "2024-02-29T12:23:17Z",
          "tree_id": "bf7cbc303127d2db62273c94dd1007e40d3ff922",
          "url": "https://github.com/equinor/ert/commit/3575519b1da29b656682fde441389deb608cbe78"
        },
        "date": 1709209584155,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.40950094121440955,
            "unit": "iter/sec",
            "range": "stddev: 0.44442149169187356",
            "extra": "mean: 2.441996829199991 sec\nrounds: 5"
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
          "id": "a69c9320a0d5b76c35abb7f7cfee4a6538f8de04",
          "message": "Rename 'analysis module'",
          "timestamp": "2024-02-29T13:33:58+01:00",
          "tree_id": "47253e9091cea7a140914f792024b0c807f13bd6",
          "url": "https://github.com/equinor/ert/commit/a69c9320a0d5b76c35abb7f7cfee4a6538f8de04"
        },
        "date": 1709210260434,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.42785895164730503,
            "unit": "iter/sec",
            "range": "stddev: 0.5273505000491615",
            "extra": "mean: 2.3372188337999886 sec\nrounds: 5"
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
          "id": "983719d049443e0ab959a1472581e9f7954b50ca",
          "message": "Use .get(key, defaultvalue) in storage_model.py (#7311)\n\nFixed by ruff",
          "timestamp": "2024-02-29T13:27:45Z",
          "tree_id": "345f10e5e705bf0286eba1aa350d93247daad6f9",
          "url": "https://github.com/equinor/ert/commit/983719d049443e0ab959a1472581e9f7954b50ca"
        },
        "date": 1709213426078,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.424722950223149,
            "unit": "iter/sec",
            "range": "stddev: 0.5026556804751776",
            "extra": "mean: 2.3544760166000005 sec\nrounds: 5"
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
          "id": "d6da218c0655a3a59fc1ca6e26dab3b5d72f67e0",
          "message": "Test all possible PBS job states (#7307)",
          "timestamp": "2024-02-29T14:31:43Z",
          "tree_id": "bc65464f2c4a53dea4f0feba4cab19520958d027",
          "url": "https://github.com/equinor/ert/commit/d6da218c0655a3a59fc1ca6e26dab3b5d72f67e0"
        },
        "date": 1709217270468,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.43520096617344123,
            "unit": "iter/sec",
            "range": "stddev: 0.4642703443352484",
            "extra": "mean: 2.297789016399997 sec\nrounds: 5"
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
          "id": "ec2bcdd2da75cde8af987fe6d2c07eef80017f9b",
          "message": "Add support for KEEP_QSUB_OUTPUT for OpenPBSDriver (#7302)\n\nThis is implemented differently compared to the legacy driver, which injects\r\nthe '-k' option to 'qsub'. Controlling using -j oe and -o and -e seems more stable",
          "timestamp": "2024-02-29T14:53:38Z",
          "tree_id": "babb790a1deea567c0e31a5649c6070dc44cbcde",
          "url": "https://github.com/equinor/ert/commit/ec2bcdd2da75cde8af987fe6d2c07eef80017f9b"
        },
        "date": 1709218582381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4287339972882231,
            "unit": "iter/sec",
            "range": "stddev: 0.49472286598541926",
            "extra": "mean: 2.3324485725999806 sec\nrounds: 5"
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
          "id": "ac990cda846e1008d95b5e26ab7c79b8b7e0b2bd",
          "message": "Enable scheduler on QUEUE_SYSTEM TORQUE by default (#7301)",
          "timestamp": "2024-02-29T15:58:23+01:00",
          "tree_id": "f18f4d3c702f5c775abe5cb1c60d2258ba514c84",
          "url": "https://github.com/equinor/ert/commit/ac990cda846e1008d95b5e26ab7c79b8b7e0b2bd"
        },
        "date": 1709218893052,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.40830435565949824,
            "unit": "iter/sec",
            "range": "stddev: 0.4906009170450564",
            "extra": "mean: 2.449153397800001 sec\nrounds: 5"
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
          "id": "c3fa53767d10bf4380ff7d7f44573dc9a26de057",
          "message": "Add warning filters for tests.\n\nthis should reduce the warning spam so that we can hopefully notice\nproblematic warnings",
          "timestamp": "2024-02-29T16:37:38+01:00",
          "tree_id": "c0457dcd8e05942c874cc9169dbdca18fd171ab7",
          "url": "https://github.com/equinor/ert/commit/c3fa53767d10bf4380ff7d7f44573dc9a26de057"
        },
        "date": 1709221223209,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4419044136982237,
            "unit": "iter/sec",
            "range": "stddev: 0.5227912400703371",
            "extra": "mean: 2.2629328176000056 sec\nrounds: 5"
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
          "id": "33e6fbd05bf7444374b91ed89bf1a4d324699b1a",
          "message": "Silence downcasting warnings in tests",
          "timestamp": "2024-03-01T07:49:07+01:00",
          "tree_id": "bcd4c394a7746dd486833684ff4267fec10a5247",
          "url": "https://github.com/equinor/ert/commit/33e6fbd05bf7444374b91ed89bf1a4d324699b1a"
        },
        "date": 1709275917176,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.42222938733482235,
            "unit": "iter/sec",
            "range": "stddev: 0.6186140823372661",
            "extra": "mean: 2.368380861200012 sec\nrounds: 5"
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
          "id": "e5288adb5a0ef8c05ab9aacbb1d59f940aa5556a",
          "message": "Remove timeouts for LSF integration tests (#7322)",
          "timestamp": "2024-03-01T07:19:23Z",
          "tree_id": "9df560a32c774419eb98a331a03d181b6e53b4ca",
          "url": "https://github.com/equinor/ert/commit/e5288adb5a0ef8c05ab9aacbb1d59f940aa5556a"
        },
        "date": 1709277730735,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.41927430631867774,
            "unit": "iter/sec",
            "range": "stddev: 0.48702979253155393",
            "extra": "mean: 2.3850734111999943 sec\nrounds: 5"
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
          "id": "59005704f0757292fa13391a1546df7c51765f01",
          "message": "Mute warnings from PBS driver (#7321)\n\nAn endless row of warnings about transition from R to E\r\nhas been observed while running on real PBS cluster",
          "timestamp": "2024-03-01T08:36:16+01:00",
          "tree_id": "21c42f434b3a5e0327b88775565507dfa06e0fce",
          "url": "https://github.com/equinor/ert/commit/59005704f0757292fa13391a1546df7c51765f01"
        },
        "date": 1709278761740,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4042925086955481,
            "unit": "iter/sec",
            "range": "stddev: 0.41160432801391045",
            "extra": "mean: 2.473456664400004 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}