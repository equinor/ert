window.BENCHMARK_DATA = {
  "lastUpdate": 1708081521202,
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
          "id": "55a5e12d72c9a25264b2a0d9c50a19554b528a95",
          "message": "Add poly example integration test for openpbs driver (#7156)",
          "timestamp": "2024-02-13T08:20:30+01:00",
          "tree_id": "1d8ed882c083c2257788c60c7a30c9a8cc8c61c8",
          "url": "https://github.com/equinor/ert/commit/55a5e12d72c9a25264b2a0d9c50a19554b528a95"
        },
        "date": 1707809003381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.483745621461996,
            "unit": "iter/sec",
            "range": "stddev: 0.030359965041432853",
            "extra": "mean: 154.23183733332735 msec\nrounds: 6"
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
          "id": "4a420d1dacb2b959db9f8c1f6c7bc3815fd1172a",
          "message": "Reduce size of grid data generation",
          "timestamp": "2024-02-13T09:32:39+01:00",
          "tree_id": "c81117a6a994531fbfc429877399833e5e902b5e",
          "url": "https://github.com/equinor/ert/commit/4a420d1dacb2b959db9f8c1f6c7bc3815fd1172a"
        },
        "date": 1707813325857,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.481839449067481,
            "unit": "iter/sec",
            "range": "stddev: 0.033143888583854045",
            "extra": "mean: 154.277193666663 msec\nrounds: 6"
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
          "id": "4cc9d21f9ff6d613712b5681420ea20abc2e81d9",
          "message": "Skip export misfit data job test on mac",
          "timestamp": "2024-02-13T10:45:05+01:00",
          "tree_id": "7e475b4fb8bf30a770e6c3c30a4c375fd52a92e8",
          "url": "https://github.com/equinor/ert/commit/4cc9d21f9ff6d613712b5681420ea20abc2e81d9"
        },
        "date": 1707817691846,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.169664877181548,
            "unit": "iter/sec",
            "range": "stddev: 0.04199446424361586",
            "extra": "mean: 162.08335783333894 msec\nrounds: 6"
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
          "id": "bf8912a2021a932629c95d50a5815875fdcb3e49",
          "message": "Add minimal Python LSF driver",
          "timestamp": "2024-02-13T12:49:40+01:00",
          "tree_id": "71967fad8d40f2c03c6b3da0b9b69c347260f14a",
          "url": "https://github.com/equinor/ert/commit/bf8912a2021a932629c95d50a5815875fdcb3e49"
        },
        "date": 1707825139388,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.522382173510748,
            "unit": "iter/sec",
            "range": "stddev: 0.030927785772699702",
            "extra": "mean: 153.31821616667676 msec\nrounds: 6"
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
          "id": "95758d0c612ee513e3485fea15cfa2858de1509d",
          "message": "Fix issue with parameter_example test\n\nthe actnum condition was set incorrectly",
          "timestamp": "2024-02-13T15:41:49+01:00",
          "tree_id": "ca78574c44251c6c03cecefcfbc6d0d1dccc9e1c",
          "url": "https://github.com/equinor/ert/commit/95758d0c612ee513e3485fea15cfa2858de1509d"
        },
        "date": 1707835460988,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.507730631918027,
            "unit": "iter/sec",
            "range": "stddev: 0.026920485113552127",
            "extra": "mean: 153.6633976666716 msec\nrounds: 6"
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
          "id": "895683d6335a7df292df46d8aa79ad0ec34bfb75",
          "message": "Enable poly-case integration test for LSF driver",
          "timestamp": "2024-02-13T17:35:14+01:00",
          "tree_id": "4a281c4b8b134afff81137a4f50486b292bf8ab1",
          "url": "https://github.com/equinor/ert/commit/895683d6335a7df292df46d8aa79ad0ec34bfb75"
        },
        "date": 1707842275182,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.185053497836678,
            "unit": "iter/sec",
            "range": "stddev: 0.03633249867026883",
            "extra": "mean: 161.68008900000075 msec\nrounds: 6"
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
          "id": "6895be5119c0769fdd5899f75d6be571459df206",
          "message": "Add logger for Azure to see usage of scheduler/job_queue (#7177)",
          "timestamp": "2024-02-14T08:32:22Z",
          "tree_id": "1eadb366b8f9b335584d108fbdf133e937b74758",
          "url": "https://github.com/equinor/ert/commit/6895be5119c0769fdd5899f75d6be571459df206"
        },
        "date": 1707899701019,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.407010988124696,
            "unit": "iter/sec",
            "range": "stddev: 0.035209477503405896",
            "extra": "mean: 156.07902060000924 msec\nrounds: 5"
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
          "id": "c8c2e0f984417a141d710b8a4a0c927ad188b76b",
          "message": "Remove unused prior-ensemble in es-mda",
          "timestamp": "2024-02-14T11:49:16+01:00",
          "tree_id": "e1b0abe9a98955d7eb9126f1846098183afe70b8",
          "url": "https://github.com/equinor/ert/commit/c8c2e0f984417a141d710b8a4a0c927ad188b76b"
        },
        "date": 1707907934498,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.495979711236443,
            "unit": "iter/sec",
            "range": "stddev: 0.027935755634504993",
            "extra": "mean: 153.94136750000106 msec\nrounds: 6"
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
          "id": "1ff65a5a5beaabe553a93682bb294b1c0a906d15",
          "message": "Remove parsing of config from dark storage",
          "timestamp": "2024-02-14T12:18:44+01:00",
          "tree_id": "44bbefd03bbd8a2096a6b3d1f8f8aea22da36b66",
          "url": "https://github.com/equinor/ert/commit/1ff65a5a5beaabe553a93682bb294b1c0a906d15"
        },
        "date": 1707909673979,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.501729210737343,
            "unit": "iter/sec",
            "range": "stddev: 0.03207370491271416",
            "extra": "mean: 153.805236666661 msec\nrounds: 6"
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
          "id": "4743427f3873d9183a140d334e3d847bcd883108",
          "message": "Remove unused format_mode\n\n* Remove unused format_mode\r\n\r\nWas always called with True",
          "timestamp": "2024-02-14T12:49:05+01:00",
          "tree_id": "b0b2751acf210fbf3fb7d43ab558f256a00b6336",
          "url": "https://github.com/equinor/ert/commit/4743427f3873d9183a140d334e3d847bcd883108"
        },
        "date": 1707911496694,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.4849243488339985,
            "unit": "iter/sec",
            "range": "stddev: 0.02795118555458032",
            "extra": "mean: 154.20380350000565 msec\nrounds: 6"
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
          "id": "5344f254f9e62b4712bc974fca5e9788eae73a7d",
          "message": "Use get function instead of if block\n\nSuggested by ruff:  SIM401 Use `counts.get(category, 0)` instead of an `if` block",
          "timestamp": "2024-02-14T15:41:20+01:00",
          "tree_id": "4a61947f9f1dad53aaeed1b9a56cd5e97a2caf43",
          "url": "https://github.com/equinor/ert/commit/5344f254f9e62b4712bc974fca5e9788eae73a7d"
        },
        "date": 1707921847107,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.377336042905381,
            "unit": "iter/sec",
            "range": "stddev: 0.03124191986536184",
            "extra": "mean: 156.80528566664975 msec\nrounds: 6"
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
          "id": "800b90208c7618ae98c81e9ac56a1c1fe21aa2ac",
          "message": "Use copy fixtures consistently",
          "timestamp": "2024-02-16T12:02:53+01:00",
          "tree_id": "2288de03e8bfe90e3f16785c577f81b1d0f4188b",
          "url": "https://github.com/equinor/ert/commit/800b90208c7618ae98c81e9ac56a1c1fe21aa2ac"
        },
        "date": 1708081520716,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.467290387026392,
            "unit": "iter/sec",
            "range": "stddev: 0.029188707157460768",
            "extra": "mean: 154.62426149999922 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}