window.BENCHMARK_DATA = {
  "lastUpdate": 1706620885199,
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "10cd8a9261298a2d042c86412beada964d4ec139",
          "message": "Add missing space in error message (#6989)",
          "timestamp": "2024-01-23T12:42:58+01:00",
          "tree_id": "e8d74754af816394e649c06d37063d69a483dcff",
          "url": "https://github.com/equinor/ert/commit/10cd8a9261298a2d042c86412beada964d4ec139"
        },
        "date": 1706010348007,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.627734997810887,
            "unit": "iter/sec",
            "range": "stddev: 0.0010747779787087122",
            "extra": "mean: 150.8811079999873 msec\nrounds: 5"
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
          "id": "aefd10a500b51c63f5cb93398dad96af4ff493b6",
          "message": "Add NaN for observations missing a response",
          "timestamp": "2024-01-23T15:08:30+01:00",
          "tree_id": "c955a8fd44f5ea9814c8067b37fca1c4d51a3ec7",
          "url": "https://github.com/equinor/ert/commit/aefd10a500b51c63f5cb93398dad96af4ff493b6"
        },
        "date": 1706019071248,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.652190223519401,
            "unit": "iter/sec",
            "range": "stddev: 0.0024028578549890414",
            "extra": "mean: 150.32642880000822 msec\nrounds: 5"
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
          "id": "433b916b70e5aa318eea3ae3e6a3eb62f723610e",
          "message": "Give proper error message when eclrun fails with nonzero return code (#6998)\n\nCatch return_code=1 properly from eclrun",
          "timestamp": "2024-01-23T21:34:00+01:00",
          "tree_id": "44a271586abe7de11ad319d41866c32e7010104f",
          "url": "https://github.com/equinor/ert/commit/433b916b70e5aa318eea3ae3e6a3eb62f723610e"
        },
        "date": 1706042202054,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.614623566918536,
            "unit": "iter/sec",
            "range": "stddev: 0.0029706218010351094",
            "extra": "mean: 151.18018280001024 msec\nrounds: 5"
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
          "id": "bb58b4685e1a1b31a95ebbaeefe4d4ffc8165a20",
          "message": "Remove unused endpoints",
          "timestamp": "2024-01-24T07:43:59+01:00",
          "tree_id": "d37c3ee1436681a7c32febdbe459707444666dce",
          "url": "https://github.com/equinor/ert/commit/bb58b4685e1a1b31a95ebbaeefe4d4ffc8165a20"
        },
        "date": 1706078799436,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7024733346461405,
            "unit": "iter/sec",
            "range": "stddev: 0.0021070097234041608",
            "extra": "mean: 149.1986539999857 msec\nrounds: 5"
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
          "id": "1da097855b1222a5587b5f16d4312d907d1b290f",
          "message": "Relax requirement on return_code from mocked eclrun (#7011)\n\nThe caught return code is apparently sometimes translated from 1 to 255,\r\nboth variants observed on RHEL7.",
          "timestamp": "2024-01-24T11:36:49+01:00",
          "tree_id": "d8ad0e268625fccc0ccf41fcf2a5400ce917356a",
          "url": "https://github.com/equinor/ert/commit/1da097855b1222a5587b5f16d4312d907d1b290f"
        },
        "date": 1706092769719,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.68729051134624,
            "unit": "iter/sec",
            "range": "stddev: 0.0023085780859735984",
            "extra": "mean: 149.53739459999724 msec\nrounds: 5"
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
          "id": "789d7dc2a790f5b4ab7026250903bf00217b9f50",
          "message": "Match observations with 1 second tolerance\n\nSince summary files have precision loss of time, we need to include some\ntolerance when matching responses to observations.\n\nAlso contains a workaround for storage not handling datetimes with\nmicroseconds due to index overflow in netcdf3.\nhttps://github.com/equinor/ert/issues/6952",
          "timestamp": "2024-01-24T12:32:33+01:00",
          "tree_id": "ef031b81304a414be2750e0388f3fa24707f391d",
          "url": "https://github.com/equinor/ert/commit/789d7dc2a790f5b4ab7026250903bf00217b9f50"
        },
        "date": 1706096119649,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6382896865103005,
            "unit": "iter/sec",
            "range": "stddev: 0.0015253361219466932",
            "extra": "mean: 150.6412113999943 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zom@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0795f09cef1d62cddfaf4ceb57cdd60a2ae4afc4",
          "message": "Scheduler: Pick last MAX_RUNNING rather than first (#7008)",
          "timestamp": "2024-01-24T12:41:57+01:00",
          "tree_id": "5fe8642980bf5da45c25ca81314442ac55d4d2e9",
          "url": "https://github.com/equinor/ert/commit/0795f09cef1d62cddfaf4ceb57cdd60a2ae4afc4"
        },
        "date": 1706096662616,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.680000053653901,
            "unit": "iter/sec",
            "range": "stddev: 0.002363945228645518",
            "extra": "mean: 149.70059759999685 msec\nrounds: 5"
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
          "id": "626631bea2b87bd724ddc351d02a2586a424c533",
          "message": "Use relative tolerance for triangular test",
          "timestamp": "2024-01-24T12:53:32+01:00",
          "tree_id": "630da90dfd5dffbea89d3aaec7d4d44b4b37ab8d",
          "url": "https://github.com/equinor/ert/commit/626631bea2b87bd724ddc351d02a2586a424c533"
        },
        "date": 1706097379600,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.658243234374008,
            "unit": "iter/sec",
            "range": "stddev: 0.0021632250751760724",
            "extra": "mean: 150.1897670000062 msec\nrounds: 5"
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
          "id": "29685e0b5305a65eff29cf03e7da9125a1e99d26",
          "message": "Test QueueConfig to Scheduler propagation\n\nAdd a test that the LegacyEnsemble object will pass the correct\narguments to the Scheduler object when it creates it.",
          "timestamp": "2024-01-24T15:54:23+01:00",
          "tree_id": "7b84b3bee3a92ebc3265d92c5c2e2858b22dea36",
          "url": "https://github.com/equinor/ert/commit/29685e0b5305a65eff29cf03e7da9125a1e99d26"
        },
        "date": 1706108220768,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.635604420305303,
            "unit": "iter/sec",
            "range": "stddev: 0.0021271414810121103",
            "extra": "mean: 150.7021722000104 msec\nrounds: 5"
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
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "1e7826fc64388033cd2fca2fa1d99ae06398a7cb",
          "message": "Update documentation on arm installation\n\nNow ert is fully arm compatible, so this is redundant.",
          "timestamp": "2024-01-25T07:57:52+01:00",
          "tree_id": "a054d97ff9bc6a1f7ff13788fafdc4846691fc3b",
          "url": "https://github.com/equinor/ert/commit/1e7826fc64388033cd2fca2fa1d99ae06398a7cb"
        },
        "date": 1706166035694,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.653486500247201,
            "unit": "iter/sec",
            "range": "stddev: 0.0022357672224931017",
            "extra": "mean: 150.2971412000079 msec\nrounds: 5"
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
          "id": "eaa61f6ed9113f95fe9c54af15cb2fe39bdddf71",
          "message": "Log when ert is used with float64 fields and surfaces\n\nLog when ert is used with float64 fields and surfaces",
          "timestamp": "2024-01-25T08:44:57+01:00",
          "tree_id": "a4732d2851488937efc28b3cd07f6bc066ae74dc",
          "url": "https://github.com/equinor/ert/commit/eaa61f6ed9113f95fe9c54af15cb2fe39bdddf71"
        },
        "date": 1706168865661,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.548011263359662,
            "unit": "iter/sec",
            "range": "stddev: 0.004612986456023278",
            "extra": "mean: 152.71812460000547 msec\nrounds: 5"
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
          "id": "b3fcd71b8987978e99e2120e709d4d697ca746d9",
          "message": "Add ContextBoolEncoder to facilitate json serialization of ContextBools.",
          "timestamp": "2024-01-25T10:28:51+02:00",
          "tree_id": "00e4ba34c874493f58e7060a304c1f8d126c6d04",
          "url": "https://github.com/equinor/ert/commit/b3fcd71b8987978e99e2120e709d4d697ca746d9"
        },
        "date": 1706171488381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7511034739391915,
            "unit": "iter/sec",
            "range": "stddev: 0.002547057986893445",
            "extra": "mean: 148.1239331999916 msec\nrounds: 5"
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
          "id": "a34969d1a4e3a583048cbb6ad3ed7fe7d88b871f",
          "message": "Remove pin for hypothesis for Python3.8",
          "timestamp": "2024-01-25T09:30:32+01:00",
          "tree_id": "fcbbc0c7e38ebdca4962329bf3ccd28372add825",
          "url": "https://github.com/equinor/ert/commit/a34969d1a4e3a583048cbb6ad3ed7fe7d88b871f"
        },
        "date": 1706171620502,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.5353143165886305,
            "unit": "iter/sec",
            "range": "stddev: 0.005337085627355793",
            "extra": "mean: 153.01482860000988 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jparu@equinor.com",
            "name": "xjules",
            "username": "xjules"
          },
          "committer": {
            "email": "jparu@equinor.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "distinct": true,
          "id": "af33945007417ed483f4201c47b586258e4f4b53",
          "message": "Enable scheduler by default for LOCAL queue\n\nThis disables scheduler for particular tests also. Additionally, we\nraise ErlCliError and ConfigValidationError when using scheduler with other Queue than LOCAL.",
          "timestamp": "2024-01-25T12:38:41+01:00",
          "tree_id": "8baa43bce7248e7dbaf5a838a40e4f4bc96fdb17",
          "url": "https://github.com/equinor/ert/commit/af33945007417ed483f4201c47b586258e4f4b53"
        },
        "date": 1706182875824,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.648831362703595,
            "unit": "iter/sec",
            "range": "stddev: 0.003032099739694689",
            "extra": "mean: 150.4023707999977 msec\nrounds: 5"
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
          "id": "52015676a5f6c9551263e3acea265dbb7b90e2da",
          "message": "Fix plotter raising ValueError when plotting same case twice (#7006)\n\n* Update plotter case selection widget\r\n\r\nChanged plotter case selectio widget to use a scrollable list of toggle\r\nbuttons to select cases instead of the previous comboboxes. This also\r\nfixes the ValueError being raised when the same case is selected twice.\r\nThis commit also adds some typing which was used in troubleshooting the\r\nbug.\r\n\r\n* Fix and add additional typing gui plot",
          "timestamp": "2024-01-25T15:50:08+01:00",
          "tree_id": "3e1aec8f64719d6753f291404b8487cb264fda12",
          "url": "https://github.com/equinor/ert/commit/52015676a5f6c9551263e3acea265dbb7b90e2da"
        },
        "date": 1706194386719,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.02264098169605,
            "unit": "iter/sec",
            "range": "stddev: 0.0440408647091672",
            "extra": "mean: 166.04011480000054 msec\nrounds: 5"
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
          "id": "41ab9c91fbf079d7bccede0d2b448739fef02294",
          "message": "Add deprecation for IES_INVERSION",
          "timestamp": "2024-01-25T16:57:12+01:00",
          "tree_id": "00b2b8f48d024be30143b01b5db1bf6b837a3a2e",
          "url": "https://github.com/equinor/ert/commit/41ab9c91fbf079d7bccede0d2b448739fef02294"
        },
        "date": 1706198397139,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.697403587815045,
            "unit": "iter/sec",
            "range": "stddev: 0.002281833927419411",
            "extra": "mean: 149.31159319999097 msec\nrounds: 5"
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
          "id": "0dfcfe8bc22c5990c274adf5276e88965501e4bc",
          "message": "Update style versions",
          "timestamp": "2024-01-26T09:50:53+01:00",
          "tree_id": "259f4dfbfbdf50d9cd047ff95cf8417acbfaeae7",
          "url": "https://github.com/equinor/ert/commit/0dfcfe8bc22c5990c274adf5276e88965501e4bc"
        },
        "date": 1706259218838,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.435439033997638,
            "unit": "iter/sec",
            "range": "stddev: 0.0027501225810322477",
            "extra": "mean: 155.3895538000006 msec\nrounds: 5"
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
          "id": "9a3ea6d040c2ca8c4fb55925ab27308fa75027a1",
          "message": "Update test_ert.yml\n\nIncrease timeout for GUI tests",
          "timestamp": "2024-01-26T12:50:40+01:00",
          "tree_id": "1d43da44f43483e2463f9bb735564aded5e48d01",
          "url": "https://github.com/equinor/ert/commit/9a3ea6d040c2ca8c4fb55925ab27308fa75027a1"
        },
        "date": 1706269990004,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.58934114370577,
            "unit": "iter/sec",
            "range": "stddev: 0.002619126930661959",
            "extra": "mean: 151.7602409999995 msec\nrounds: 5"
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
          "id": "6ee7294da6f1ba3734e99480505b19cabf1423cd",
          "message": "Catch OSError so that Ctrl-c works for ERT\n\n_base_service.py registers an interrupt handler that translates\nctrl-c to an OSError. Since this is not caught, the main thread\ndies, but the remaining threads continue (but inherits the\nsame interrupt handler).\n\nCo-authored-by: Sondre Sortland <sondreso@users.noreply.github.com>",
          "timestamp": "2024-01-26T15:11:26+01:00",
          "tree_id": "10c1b6546758172c387858857321a048d3c18c5b",
          "url": "https://github.com/equinor/ert/commit/6ee7294da6f1ba3734e99480505b19cabf1423cd"
        },
        "date": 1706278455054,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.248919124458768,
            "unit": "iter/sec",
            "range": "stddev: 0.00593236271655765",
            "extra": "mean: 160.02767520000702 msec\nrounds: 5"
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
          "id": "a9181591b145fd091534f77c687fcb7e30905631",
          "message": "Update docstring of load_all_gen_kw_data\n\nDon't calculate ens_mask if realization_index is not None.",
          "timestamp": "2024-01-29T13:18:24+01:00",
          "tree_id": "9ca710658cedaa47eb18bdc6e8409f458d443bf2",
          "url": "https://github.com/equinor/ert/commit/a9181591b145fd091534f77c687fcb7e30905631"
        },
        "date": 1706530882198,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6643049347412155,
            "unit": "iter/sec",
            "range": "stddev: 0.0026793384991423983",
            "extra": "mean: 150.05315779999364 msec\nrounds: 5"
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
          "id": "a90b11a0ecc25aa73ea8f7549d5aeaf78002ccae",
          "message": "Remove mention of opencensus.\n\nOpencensus was replaced with Opentelemetry and there is no further reason to exclude `opencensus.ext.azure.common.transport` records when capturing logs.",
          "timestamp": "2024-01-29T15:59:00+02:00",
          "tree_id": "8447805122c44d917fa1f390b3d0886226d974e5",
          "url": "https://github.com/equinor/ert/commit/a90b11a0ecc25aa73ea8f7549d5aeaf78002ccae"
        },
        "date": 1706536888740,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.700251480672506,
            "unit": "iter/sec",
            "range": "stddev: 0.002496472154968146",
            "extra": "mean: 149.2481294000072 msec\nrounds: 5"
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
          "id": "0c4a70cd12abb2b19d743616163131c067fbe4bf",
          "message": "Check scheduler loop is still running before killing all the jobs.",
          "timestamp": "2024-01-29T16:53:45+02:00",
          "tree_id": "d8e116f61749fbcc93b91e30542909da4ba90606",
          "url": "https://github.com/equinor/ert/commit/0c4a70cd12abb2b19d743616163131c067fbe4bf"
        },
        "date": 1706540177244,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.537034919594398,
            "unit": "iter/sec",
            "range": "stddev: 0.002100702794487692",
            "extra": "mean: 152.9745538000043 msec\nrounds: 5"
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
          "id": "6b784ef056a9d3f09b284aa3557f67f238386c50",
          "message": "Move experiment creation",
          "timestamp": "2024-01-30T10:19:02+01:00",
          "tree_id": "4ed139c04c9d451cef3c4a3da638eb3a2c8c1001",
          "url": "https://github.com/equinor/ert/commit/6b784ef056a9d3f09b284aa3557f67f238386c50"
        },
        "date": 1706606502221,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.401245622994807,
            "unit": "iter/sec",
            "range": "stddev: 0.006792502901449424",
            "extra": "mean: 156.21959519999677 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "1a5b6a6be9cb7fda0b99a12088b25bf5a548f2c0",
          "message": "Do not close main asyncio event loop of ert in scheduler",
          "timestamp": "2024-01-30T10:32:23+01:00",
          "tree_id": "d0a5746c30783e8c932b5e4bad658aeab546014e",
          "url": "https://github.com/equinor/ert/commit/1a5b6a6be9cb7fda0b99a12088b25bf5a548f2c0"
        },
        "date": 1706607303802,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7133615423858055,
            "unit": "iter/sec",
            "range": "stddev: 0.002594944457024593",
            "extra": "mean: 148.95667300000923 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "64ad91a55b9d470dbc8e0681669057ab312cb971",
          "message": "Add documentation for queue behavior on early exit",
          "timestamp": "2024-01-30T10:32:37+01:00",
          "tree_id": "41872aab1ebb263dd1974a7067c4a3891cef6d3e",
          "url": "https://github.com/equinor/ert/commit/64ad91a55b9d470dbc8e0681669057ab312cb971"
        },
        "date": 1706607312535,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.703020228756857,
            "unit": "iter/sec",
            "range": "stddev: 0.0026208204216903873",
            "extra": "mean: 149.18648099999245 msec\nrounds: 5"
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
          "id": "055e4fd0ec724caa35b178591bd1342fdc456089",
          "message": "Add TorqueDriver",
          "timestamp": "2024-01-30T10:39:32+01:00",
          "tree_id": "1253392d364d0f9a7b262dd685f5207bceadc95b",
          "url": "https://github.com/equinor/ert/commit/055e4fd0ec724caa35b178591bd1342fdc456089"
        },
        "date": 1706607724829,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.065828346677965,
            "unit": "iter/sec",
            "range": "stddev: 0.0317900236936553",
            "extra": "mean: 164.85794566667286 msec\nrounds: 6"
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
          "id": "9f74a17d64076f5e4164b3bd323ce9bf5b5f2fa1",
          "message": "Replace observation_config with observations",
          "timestamp": "2024-01-30T11:21:04+01:00",
          "tree_id": "fdf746e0a9de332a92fee6f7adc514441c7fc723",
          "url": "https://github.com/equinor/ert/commit/9f74a17d64076f5e4164b3bd323ce9bf5b5f2fa1"
        },
        "date": 1706610213865,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.636720676853125,
            "unit": "iter/sec",
            "range": "stddev: 0.0008689042703591679",
            "extra": "mean: 150.67682499998796 msec\nrounds: 5"
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
          "id": "0e6a770eacda8180aa2f5ba2196b98985ebf78f0",
          "message": "Fix a bug in migration",
          "timestamp": "2024-01-30T11:27:40+01:00",
          "tree_id": "e19f7a4799bc89c7001e40fc05927ca4cc903519",
          "url": "https://github.com/equinor/ert/commit/0e6a770eacda8180aa2f5ba2196b98985ebf78f0"
        },
        "date": 1706610613261,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.177102844688486,
            "unit": "iter/sec",
            "range": "stddev: 0.031197179684927155",
            "extra": "mean: 161.88819016666875 msec\nrounds: 6"
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
          "id": "48d428aaedbb0fd64dc5653e663b05d7421b0c95",
          "message": "Remove unused ignore_current from CaseSelector",
          "timestamp": "2024-01-30T11:53:34+01:00",
          "tree_id": "5c61566e121e94572b9bf02ef177daf02ec567e4",
          "url": "https://github.com/equinor/ert/commit/48d428aaedbb0fd64dc5653e663b05d7421b0c95"
        },
        "date": 1706612177458,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.69648602966981,
            "unit": "iter/sec",
            "range": "stddev: 0.002181149095666006",
            "extra": "mean: 149.33205200001112 msec\nrounds: 5"
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
          "id": "fd17a0fd58cebebfbe5c812f2a2f2c58c2295dbe",
          "message": "Remove libres from remaining endpoints",
          "timestamp": "2024-01-30T14:18:40+01:00",
          "tree_id": "8ea54e29752f0af31f80154069993af46da363c1",
          "url": "https://github.com/equinor/ert/commit/fd17a0fd58cebebfbe5c812f2a2f2c58c2295dbe"
        },
        "date": 1706620884741,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.974379256917177,
            "unit": "iter/sec",
            "range": "stddev: 0.03417923555673988",
            "extra": "mean: 167.3814059999946 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}