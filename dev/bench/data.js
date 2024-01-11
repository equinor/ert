window.BENCHMARK_DATA = {
  "lastUpdate": 1704963317744,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "9df8d9ca0e9d54120248e53e86906efe5ea3163d",
          "message": "Fix a bug that caused the plot to crash ert",
          "timestamp": "2024-01-03T10:41:10+01:00",
          "tree_id": "95f07a13216fce06cee5edb43f5f54239f1782f0",
          "url": "https://github.com/equinor/ert/commit/9df8d9ca0e9d54120248e53e86906efe5ea3163d"
        },
        "date": 1704275015791,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.201083912261648,
            "unit": "iter/sec",
            "range": "stddev: 0.029587425456725467",
            "extra": "mean: 161.26212999999248 msec\nrounds: 6"
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
          "id": "d8b53bb25053680d09095960d36fc164f6f9b00b",
          "message": "Avoid crashing due to empty case in plotter configuration",
          "timestamp": "2024-01-03T12:09:24+01:00",
          "tree_id": "7ed785539f5cc4f9b51ebb6e9a0c5b6c0d3502a3",
          "url": "https://github.com/equinor/ert/commit/d8b53bb25053680d09095960d36fc164f6f9b00b"
        },
        "date": 1704280325004,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.109143352643157,
            "unit": "iter/sec",
            "range": "stddev: 0.033512906177039586",
            "extra": "mean: 163.68907100000266 msec\nrounds: 6"
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
          "id": "f5ad501ce188525cdc104baed10768f9ac96cacd",
          "message": "Fix roff outputting double for parameters",
          "timestamp": "2024-01-03T13:09:33+01:00",
          "tree_id": "9793f4405b7bc038eb01a3061f03dcbfb8604fef",
          "url": "https://github.com/equinor/ert/commit/f5ad501ce188525cdc104baed10768f9ac96cacd"
        },
        "date": 1704283937149,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.162590933586845,
            "unit": "iter/sec",
            "range": "stddev: 0.03129532681757613",
            "extra": "mean: 162.26941083333676 msec\nrounds: 6"
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
          "id": "cfdb5c38c322b66131c69db7938300278a2cf2a3",
          "message": "Make suggestor work correctly with resizing",
          "timestamp": "2024-01-03T14:13:36+01:00",
          "tree_id": "5848e0c6f8ef9b213f349411321dc5f05b4a6f69",
          "url": "https://github.com/equinor/ert/commit/cfdb5c38c322b66131c69db7938300278a2cf2a3"
        },
        "date": 1704287766509,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.063075563532016,
            "unit": "iter/sec",
            "range": "stddev: 0.034645634778718105",
            "extra": "mean: 164.93279516665874 msec\nrounds: 6"
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
          "id": "6d2f9adcbefea7c0a28777d5e49f43cbe121f9d8",
          "message": "Add tests for local driver (#6843)\n\n* Add unit test for local driver\r\n\r\nAdded local driver unit test for testing that killing an already dead job does not raise an exception.\r\n\r\n* Add local driver integration test for orphaned processes\r\n\r\nThe test added makes sure processes started by ERT are not killed in the event of ERT unexpectedly closing\r\n\r\n* Have local driver integration test run on scheduler and queue",
          "timestamp": "2024-01-03T14:36:41+01:00",
          "tree_id": "58912dc4bef21252e91fc2617dae5aa6e8d273f4",
          "url": "https://github.com/equinor/ert/commit/6d2f9adcbefea7c0a28777d5e49f43cbe121f9d8"
        },
        "date": 1704289149964,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.1878857578257,
            "unit": "iter/sec",
            "range": "stddev: 0.029221431018367137",
            "extra": "mean: 161.60608633333595 msec\nrounds: 6"
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
          "id": "ba865c9afc8cd65b588de3f05f4242c05176d8e5",
          "message": "Add stop_long_running_jobs funcitonality to Scheduler\n\nThis adds two tasks to scheduler. 1) Processing the finished jobs and computing the running average 2) Checking that the duration of still running jobs is bellow the threshold and kills those jobs otherwise.",
          "timestamp": "2024-01-03T15:33:31+01:00",
          "tree_id": "6fe26b4484db5e5ac7785ac4da5903d000f787e2",
          "url": "https://github.com/equinor/ert/commit/ba865c9afc8cd65b588de3f05f4242c05176d8e5"
        },
        "date": 1704292562620,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.968291214812013,
            "unit": "iter/sec",
            "range": "stddev: 0.036683280106644275",
            "extra": "mean: 167.5521458333359 msec\nrounds: 6"
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
          "id": "ef62f37229b9060b008bb47c74f5198ee4adf69d",
          "message": "Clarify fixture param for try_queue_and_scheduler (#6884)\n\nNow, when a test using the try_queue_and_scheduler-fixture will have the ids \"using_job_queue\" or \"using_scheduler\" instead of True or False. This will make it easier to see whether the test is failing with scheduler or job queue.",
          "timestamp": "2024-01-03T14:47:31Z",
          "tree_id": "016d8cdadb84becf81a706cea55efd336c7ae915",
          "url": "https://github.com/equinor/ert/commit/ef62f37229b9060b008bb47c74f5198ee4adf69d"
        },
        "date": 1704293423378,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.136463654445345,
            "unit": "iter/sec",
            "range": "stddev: 0.03212229716539486",
            "extra": "mean: 162.9603068333315 msec\nrounds: 6"
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
          "id": "b89d49d3dc84e49fb03a05e8042d0c9636db9c6f",
          "message": "Merge checklist sections",
          "timestamp": "2024-01-03T15:54:47+01:00",
          "tree_id": "2fdc28a7a3af9d68cfe1ca5bd97ae31195b41ddf",
          "url": "https://github.com/equinor/ert/commit/b89d49d3dc84e49fb03a05e8042d0c9636db9c6f"
        },
        "date": 1704293848052,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.710758173523607,
            "unit": "iter/sec",
            "range": "stddev: 0.03709633629943954",
            "extra": "mean: 175.10809766665147 msec\nrounds: 6"
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
          "id": "6d0948cf24f052c5492e17ef00e4855116246687",
          "message": "Add missing __init__ file",
          "timestamp": "2024-01-04T09:51:57+01:00",
          "tree_id": "0b7a26585e23fd0f8f640446a593516401f7a208",
          "url": "https://github.com/equinor/ert/commit/6d0948cf24f052c5492e17ef00e4855116246687"
        },
        "date": 1704358495256,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.209411476854901,
            "unit": "iter/sec",
            "range": "stddev: 0.02983067403756598",
            "extra": "mean: 161.04585816665917 msec\nrounds: 6"
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
          "id": "533fef707de7ae202cdc2ccfb27e351d2c849753",
          "message": "Add deprecation warning to functions moved from facade",
          "timestamp": "2024-01-04T09:59:24+01:00",
          "tree_id": "1a8309688bb4ff695244bbc16bd3b59a02eebcdf",
          "url": "https://github.com/equinor/ert/commit/533fef707de7ae202cdc2ccfb27e351d2c849753"
        },
        "date": 1704358928716,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.145316593078664,
            "unit": "iter/sec",
            "range": "stddev: 0.031051692724299213",
            "extra": "mean: 162.7255463333294 msec\nrounds: 6"
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
          "id": "9123159b0a56d7c302c2ead1b4cd789ade995e1e",
          "message": "Fix unit test scheduler local driver (#6894)\n\nThis commit fixes unit_tests/scheduler/test_local_driver.py::test_that_killing_killed_job_does_not_raise by changing the stub process to sleep 10 instead of 5. This is an exact copy of the test_kill test above it. The bug was that the process was already done by the time we tried killing it.",
          "timestamp": "2024-01-04T10:13:20+01:00",
          "tree_id": "a6edb51160347f123bf7e325b256132340698d31",
          "url": "https://github.com/equinor/ert/commit/9123159b0a56d7c302c2ead1b4cd789ade995e1e"
        },
        "date": 1704359795712,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 4.984417117078171,
            "unit": "iter/sec",
            "range": "stddev: 0.04633484528145767",
            "extra": "mean: 200.62526400001465 msec\nrounds: 6"
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
          "id": "a762b9b9e9d3a35ae3e90b58e553685a1b14977d",
          "message": "Remove the state_map for ensembles",
          "timestamp": "2024-01-04T13:25:01+01:00",
          "tree_id": "e92b5970719e877e3dc5c1c4e9d8681cbcf74da8",
          "url": "https://github.com/equinor/ert/commit/a762b9b9e9d3a35ae3e90b58e553685a1b14977d"
        },
        "date": 1704371269758,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.122185025957407,
            "unit": "iter/sec",
            "range": "stddev: 0.030427528174608264",
            "extra": "mean: 163.340375333334 msec\nrounds: 6"
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
          "id": "a79a20b5ce99df9862307435462da0057ca8a395",
          "message": "Use custom loop to get additional exceptions\n\n - Monkeypatch forward_modek_ok for scheduler",
          "timestamp": "2024-01-04T13:30:08+01:00",
          "tree_id": "ce36b0ef6f3450e72920798da66ed00e62852493",
          "url": "https://github.com/equinor/ert/commit/a79a20b5ce99df9862307435462da0057ca8a395"
        },
        "date": 1704371572872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.036899236576398,
            "unit": "iter/sec",
            "range": "stddev: 0.03386776503935381",
            "extra": "mean: 165.64795283333447 msec\nrounds: 6"
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
          "id": "d50968562bf5207358789a3aa454b1a9759cdab6",
          "message": "Add unit tests for scheduler job states (#6824)\n\n* Add unit tests for scheduler job",
          "timestamp": "2024-01-05T08:52:13+01:00",
          "tree_id": "481494ce62f83ffb855318e321a953e15cbc3421",
          "url": "https://github.com/equinor/ert/commit/d50968562bf5207358789a3aa454b1a9759cdab6"
        },
        "date": 1704441287253,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.859968327516018,
            "unit": "iter/sec",
            "range": "stddev: 0.029343204471406484",
            "extra": "mean: 170.64938649999326 msec\nrounds: 6"
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
          "id": "a8192b71aab34f62cdc2bc5d8888519319e735cc",
          "message": "Remove duplicate logging in scheduler job (#6902)",
          "timestamp": "2024-01-05T09:04:46+01:00",
          "tree_id": "cc9614c3694988af89220fb234b99337854678e1",
          "url": "https://github.com/equinor/ert/commit/a8192b71aab34f62cdc2bc5d8888519319e735cc"
        },
        "date": 1704442043654,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.1125446862164,
            "unit": "iter/sec",
            "range": "stddev: 0.0308779583743804",
            "extra": "mean: 163.5979859999992 msec\nrounds: 6"
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
          "id": "e5ee36dab32477daa0b502227fb0c9a001f5e5ec",
          "message": "Support max_running in Scheduler",
          "timestamp": "2024-01-05T09:33:52+01:00",
          "tree_id": "698286524d2011a712ca0603d0d9bfa159da7f5c",
          "url": "https://github.com/equinor/ert/commit/e5ee36dab32477daa0b502227fb0c9a001f5e5ec"
        },
        "date": 1704443814172,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.151327150678222,
            "unit": "iter/sec",
            "range": "stddev: 0.029086009791191407",
            "extra": "mean: 162.56654466666495 msec\nrounds: 6"
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
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "41460e0d71eb1c40c196142a646c85ed2f0f7d7d",
          "message": "Extract copy of non-updated params to func",
          "timestamp": "2024-01-05T11:51:48+01:00",
          "tree_id": "79e4bc57cb45834a5453e47249064a9738aa3afc",
          "url": "https://github.com/equinor/ert/commit/41460e0d71eb1c40c196142a646c85ed2f0f7d7d"
        },
        "date": 1704452077383,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.2014655092441915,
            "unit": "iter/sec",
            "range": "stddev: 0.029228908023129052",
            "extra": "mean: 161.25220699999923 msec\nrounds: 6"
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
          "id": "f346b4e6d64f5b822c131c997bf0a436390cf39c",
          "message": "Make sure to log exception only when there is one",
          "timestamp": "2024-01-05T12:17:12+01:00",
          "tree_id": "4a630a26fb10ad7f4af190e623b9e3adffbcc594",
          "url": "https://github.com/equinor/ert/commit/f346b4e6d64f5b822c131c997bf0a436390cf39c"
        },
        "date": 1704453601798,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.1002010111805065,
            "unit": "iter/sec",
            "range": "stddev: 0.031843213497262796",
            "extra": "mean: 163.9290243333278 msec\nrounds: 6"
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
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "8c22f231c6f7fcb4fecdba5c6cd183e07b1e7e67",
          "message": "Reduce accuracy of surface-order test\n\nTest is flaky.\nTest makes sure ERT does not change order of arrays and numerical\naccuracy is not important.",
          "timestamp": "2024-01-05T14:04:05+01:00",
          "tree_id": "2782dbc633aa2483bc366a72d5be27ae56e9971c",
          "url": "https://github.com/equinor/ert/commit/8c22f231c6f7fcb4fecdba5c6cd183e07b1e7e67"
        },
        "date": 1704460029143,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.722562605385406,
            "unit": "iter/sec",
            "range": "stddev: 0.038534614854017504",
            "extra": "mean: 174.7468868333423 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "committer": {
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "9b92e9ad9407fd2b81d91e2e3a1570f64fa4faeb",
          "message": "Update README.md",
          "timestamp": "2024-01-05T14:14:29+01:00",
          "tree_id": "7507a7b6d30f17df70b5f2615f7af609b778fad6",
          "url": "https://github.com/equinor/ert/commit/9b92e9ad9407fd2b81d91e2e3a1570f64fa4faeb"
        },
        "date": 1704460623496,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.042098326129351,
            "unit": "iter/sec",
            "range": "stddev: 0.03499003155235332",
            "extra": "mean: 165.50541650000147 msec\nrounds: 6"
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
          "id": "0a64025cf795e50d9362840dfe9b6a0e4c3c3f2a",
          "message": "Ignore CancelledError when killing (#6909)\n\nAs this kill() is called from job.py when a CancelledError is\r\nhandled, some more exception handling is required here. Solves\r\na flaky test",
          "timestamp": "2024-01-08T17:03:26+01:00",
          "tree_id": "98d082ed73a13ecf33b312ac211c6304530c067b",
          "url": "https://github.com/equinor/ert/commit/0a64025cf795e50d9362840dfe9b6a0e4c3c3f2a"
        },
        "date": 1704729959518,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.995391330693898,
            "unit": "iter/sec",
            "range": "stddev: 0.03291803311682125",
            "extra": "mean: 166.79478366665043 msec\nrounds: 6"
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
          "id": "ca45684edd357134272493c0dcc74eb5f992daa3",
          "message": "Append to update_log\n\nupdate_log shall show all updated observations.\nCurrent implementation overwrites the update_log\nfor each update step.\nEach update step needs a unique name in order for this to work.",
          "timestamp": "2024-01-09T08:36:38+01:00",
          "tree_id": "dbbf4c508b30f3373ff19b8bd9cb254c63717228",
          "url": "https://github.com/equinor/ert/commit/ca45684edd357134272493c0dcc74eb5f992daa3"
        },
        "date": 1704785959188,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.976708426589098,
            "unit": "iter/sec",
            "range": "stddev: 0.0364303888841622",
            "extra": "mean: 167.31617616666958 msec\nrounds: 6"
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
          "id": "c39e691b850d66fc59b4f7c6427791f167e12a19",
          "message": "Remove unused function _split_by",
          "timestamp": "2024-01-09T12:51:12+01:00",
          "tree_id": "26c0bfdb6f73c44951b086a23a7ad29dc028ab62",
          "url": "https://github.com/equinor/ert/commit/c39e691b850d66fc59b4f7c6427791f167e12a19"
        },
        "date": 1704801228660,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.1799624087716065,
            "unit": "iter/sec",
            "range": "stddev: 0.026897780650949575",
            "extra": "mean: 161.81328200000658 msec\nrounds: 6"
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
          "id": "241a9db16d045bd6642dc8b3c116f2c4d03d293d",
          "message": "Add scheduler logs to jobqeue_file handler",
          "timestamp": "2024-01-09T15:12:00+01:00",
          "tree_id": "a0af922bd2d121da3098f37e9e89bffaa75bb93d",
          "url": "https://github.com/equinor/ert/commit/241a9db16d045bd6642dc8b3c116f2c4d03d293d"
        },
        "date": 1704809701788,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.091560658358257,
            "unit": "iter/sec",
            "range": "stddev: 0.029999220679694825",
            "extra": "mean: 164.1615435000053 msec\nrounds: 6"
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
          "id": "881a350c2fef60983a1042309d82ad39aeac8178",
          "message": "Allow feature toggling from environment variables\n\nadd possibility for optional toggles",
          "timestamp": "2024-01-09T21:36:47+01:00",
          "tree_id": "4fe9d568355fa3b86c93d0b565c9cc15845daa5d",
          "url": "https://github.com/equinor/ert/commit/881a350c2fef60983a1042309d82ad39aeac8178"
        },
        "date": 1704832763501,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.038990737659192,
            "unit": "iter/sec",
            "range": "stddev: 0.033412894441354216",
            "extra": "mean: 165.5905834999866 msec\nrounds: 6"
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
          "id": "86d4550e8b3a9c0f0a4f59d5632f0ae0c3447e6b",
          "message": "Upload coverage for doctests",
          "timestamp": "2024-01-10T08:16:31+01:00",
          "tree_id": "b333cea37561591a568a8e0d368ce57d511b2aa0",
          "url": "https://github.com/equinor/ert/commit/86d4550e8b3a9c0f0a4f59d5632f0ae0c3447e6b"
        },
        "date": 1704871140156,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.879516794851588,
            "unit": "iter/sec",
            "range": "stddev: 0.043641340672242346",
            "extra": "mean: 170.08200416667782 msec\nrounds: 6"
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
          "id": "18ac76c803dafa3a6ed75c47584a5646b682ac0c",
          "message": "Added tests for missing keywords",
          "timestamp": "2024-01-10T12:28:45+01:00",
          "tree_id": "8727548ec0c751f770c8955b807f6a8922f785ae",
          "url": "https://github.com/equinor/ert/commit/18ac76c803dafa3a6ed75c47584a5646b682ac0c"
        },
        "date": 1704886295453,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.136811253284065,
            "unit": "iter/sec",
            "range": "stddev: 0.03159115898751113",
            "extra": "mean: 162.95107650000773 msec\nrounds: 6"
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
          "id": "df1ca32db784004dbb78f8ab9abc337bdb446975",
          "message": "Remove unused queue.stop_jobs and rename JobQueue.stop_jobs_async -> JobQueue.stop_jobs",
          "timestamp": "2024-01-10T13:34:18+01:00",
          "tree_id": "8181e2b12b8d38187edb355cb7995d008bab6c88",
          "url": "https://github.com/equinor/ert/commit/df1ca32db784004dbb78f8ab9abc337bdb446975"
        },
        "date": 1704890239646,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.960678506617569,
            "unit": "iter/sec",
            "range": "stddev: 0.03364176252034566",
            "extra": "mean: 167.7661358333277 msec\nrounds: 6"
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
          "id": "f27f7f88c9824763232c2e7ced151f6fd8bbfbfe",
          "message": "Ensure consistent log levels in integration tests (#6922)\n\nSome tests were moved from unit_tests to integration_tests in /a810eb0f8715e182d4e1b1dc1636356b97023711\r\nthat depend on log level details. This dependency was not triggered in Github actions workflows\r\nbut in komodo bleeding nightly tests.",
          "timestamp": "2024-01-10T14:28:43Z",
          "tree_id": "d1dd3232f8bb2e1af899d79cd4476bf859c15a23",
          "url": "https://github.com/equinor/ert/commit/f27f7f88c9824763232c2e7ced151f6fd8bbfbfe"
        },
        "date": 1704897070178,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.736996998894375,
            "unit": "iter/sec",
            "range": "stddev: 0.002362116927727187",
            "extra": "mean: 148.43408719999616 msec\nrounds: 5"
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
          "id": "a1fce20ada3601a6d6607a61d6178b8ac525aca0",
          "message": "Set max items in benchmark chart",
          "timestamp": "2024-01-11T09:52:47+01:00",
          "tree_id": "44a3275246a085156783c39cb1d83905a422a9cb",
          "url": "https://github.com/equinor/ert/commit/a1fce20ada3601a6d6607a61d6178b8ac525aca0"
        },
        "date": 1704963317289,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.032876152560895,
            "unit": "iter/sec",
            "range": "stddev: 0.03535538852803025",
            "extra": "mean: 165.7584168333225 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}