window.BENCHMARK_DATA = {
  "lastUpdate": 1709559122255,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "5ac8d3f5bf14fc423ed221d4a5c9a3aab1686067",
          "message": "Remove RowScaling\n\nThis was an experimental feature that was never used in production,\nonly in testing. We decided to go another path when it comes to\nimplementing distance based localization, so removing this.",
          "timestamp": "2024-03-01T08:37:40+01:00",
          "tree_id": "45b5102491e4fa9571f65463663883e35413794a",
          "url": "https://github.com/equinor/ert/commit/5ac8d3f5bf14fc423ed221d4a5c9a3aab1686067"
        },
        "date": 1709278825637,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4115023815620079,
            "unit": "iter/sec",
            "range": "stddev: 0.46403284216903784",
            "extra": "mean: 2.4301195930000064 sec\nrounds: 5"
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
          "id": "f26a69942191141f1d29998f2fb4010655096502",
          "message": "Remove UpdateConfiguration\n\nThis was part of an experimental implementation of distance based\nlocalization, however it has problems in that it allowed the user\nto configure ert to update the same parameter multiple times.\nAdditionally it was not used in production, only in testing, and as\nsuch added a lot of unneeded complexity.",
          "timestamp": "2024-03-01T08:57:50+01:00",
          "tree_id": "b9b1159186fcb11511ef5e883ba756e75401b05f",
          "url": "https://github.com/equinor/ert/commit/f26a69942191141f1d29998f2fb4010655096502"
        },
        "date": 1709280052027,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19438542185214822,
            "unit": "iter/sec",
            "range": "stddev: 0.03634351675874752",
            "extra": "mean: 5.144418704200007 sec\nrounds: 5"
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
          "id": "a3c25e51f4bd2b164eeaee295f35848d58de2806",
          "message": "Remove unneeded jobname",
          "timestamp": "2024-03-01T09:59:58+01:00",
          "tree_id": "006c2d469c53b07de7e302f46ab3245ea0efc8d4",
          "url": "https://github.com/equinor/ert/commit/a3c25e51f4bd2b164eeaee295f35848d58de2806"
        },
        "date": 1709283777332,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18881468458567643,
            "unit": "iter/sec",
            "range": "stddev: 0.04688755847555915",
            "extra": "mean: 5.296198239000001 sec\nrounds: 5"
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
          "id": "0a4b5c5050dcf634e7ea4d1c28cfdbe90568768c",
          "message": "Pin pytest due to flaky\n\nhttps://github.com/box/flaky/issues/198",
          "timestamp": "2024-03-04T07:54:40+01:00",
          "tree_id": "babdde3945b36db64c73f929a9f8c1f99175f6a4",
          "url": "https://github.com/equinor/ert/commit/0a4b5c5050dcf634e7ea4d1c28cfdbe90568768c"
        },
        "date": 1709535484949,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18180068553175602,
            "unit": "iter/sec",
            "range": "stddev: 0.0718576727359095",
            "extra": "mean: 5.500529313600003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mtha@equinor.com",
            "name": "Matt Hall",
            "username": "kwinkunks"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a8053571810a1942ed852c9e47a8c93864236fda",
          "message": "Fix typos in ENKF_ALPHA docs (#6878)\n\n* Fix typos in ENKF_ALPHA docs ￼…\r\n\r\nUsing simpler and more consistent LaTeX with upright bold for vectors,\r\nwhich is a typical convention. Fixes equinor#6877.\r\n\r\n* Update docs/reference/configuration/keywords.rst\r\n\r\nCo-authored-by: Feda Curic <feda.curic@gmail.com>\r\n\r\n* Revisit the notation\r\n\r\nThe subscripts are awkward, but I think this is consistent.\r\n\r\n* Switch order of embellishments\r\n\r\nShould be the same result, but meaning is clearer\r\n\r\n---------\r\n\r\nCo-authored-by: Feda Curic <feda.curic@gmail.com>",
          "timestamp": "2024-03-04T11:43:03+01:00",
          "tree_id": "57138e4c7d27fbb093049f96ab7ef2a781338649",
          "url": "https://github.com/equinor/ert/commit/a8053571810a1942ed852c9e47a8c93864236fda"
        },
        "date": 1709549177872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1914724128330193,
            "unit": "iter/sec",
            "range": "stddev: 0.05718233767800734",
            "extra": "mean: 5.222684486000015 sec\nrounds: 5"
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
          "id": "2a245c46190272a40e227a71724e0a6c69ac8dac",
          "message": "Mute transitions from PBS (#7334)\n\nAvoiding WARNING as logs from INFO and above are kept\r\ncentrally.",
          "timestamp": "2024-03-04T12:06:10+01:00",
          "tree_id": "b14f626f70caaaa97003141108e734fc475412fc",
          "url": "https://github.com/equinor/ert/commit/2a245c46190272a40e227a71724e0a6c69ac8dac"
        },
        "date": 1709550573613,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1907470851556616,
            "unit": "iter/sec",
            "range": "stddev: 0.04247751940240171",
            "extra": "mean: 5.242544069199994 sec\nrounds: 5"
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
          "id": "c385d319d312c2b0119937e134fc17b80a7dfd4a",
          "message": "Run LSF integration tests during Komodo tests (#7328)\n\nThis requires a real LSF cluster up and running and a \"bsub\"\r\ncommand in PATH.",
          "timestamp": "2024-03-04T12:07:24+01:00",
          "tree_id": "97907aa7fdc7d56ee24f9913a7f86cec6139728a",
          "url": "https://github.com/equinor/ert/commit/c385d319d312c2b0119937e134fc17b80a7dfd4a"
        },
        "date": 1709550632776,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19045834795361,
            "unit": "iter/sec",
            "range": "stddev: 0.04074040003150719",
            "extra": "mean: 5.250491830599993 sec\nrounds: 5"
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
          "id": "ae1bdd5bbde2e96750bc776bef8cbe0782110377",
          "message": "Combine OpenPBS & LSF integration tests\n\nThere is a lot of overlap between the two drivers, so we combine some of\nthe generic integration tests into a single parameterised pytest.",
          "timestamp": "2024-03-04T14:20:52+01:00",
          "tree_id": "fd8d2b9d2be25986c3ce1d4b5941ac483b6e2cde",
          "url": "https://github.com/equinor/ert/commit/ae1bdd5bbde2e96750bc776bef8cbe0782110377"
        },
        "date": 1709558646023,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1869188932267656,
            "unit": "iter/sec",
            "range": "stddev: 0.06791015381687718",
            "extra": "mean: 5.349913979999997 sec\nrounds: 5"
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
          "id": "d45b78400ff8bc4e0bf8ec9fc61b5ea26a934a53",
          "message": "Upgrade setup python to v5 in github workflows",
          "timestamp": "2024-03-04T14:28:50+01:00",
          "tree_id": "2cd86dd7608571cdd99024f668e042d818a41b89",
          "url": "https://github.com/equinor/ert/commit/d45b78400ff8bc4e0bf8ec9fc61b5ea26a934a53"
        },
        "date": 1709559121263,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19136024153549663,
            "unit": "iter/sec",
            "range": "stddev: 0.04219778951530669",
            "extra": "mean: 5.22574591240001 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}