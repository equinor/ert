window.BENCHMARK_DATA = {
  "lastUpdate": 1710340089429,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "10550a6e93a41053c5856e7093bd54577acd8bc1",
          "message": "Autodetect num cpus/cores in pytest xdist",
          "timestamp": "2024-03-07T09:56:04+01:00",
          "tree_id": "d65209f7f7f5fd0a69562c16ebaa5b6c50e99cc3",
          "url": "https://github.com/equinor/ert/commit/10550a6e93a41053c5856e7093bd54577acd8bc1"
        },
        "date": 1709801939129,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1922751376347079,
            "unit": "iter/sec",
            "range": "stddev: 0.034169463093655365",
            "extra": "mean: 5.200880427400034 sec\nrounds: 5"
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
          "id": "f8ede6f539dec5f04be18ce0ec3835ebb5c6edd9",
          "message": "OpenPBS: Treat 'E' state the same as 'F'\n\nBoth E and F mean that the process has finished. The difference seems\nthat jobs in E state might still be using the queue. This distinction is\nof no concern to us.",
          "timestamp": "2024-03-07T14:32:01+01:00",
          "tree_id": "79022cfd64b6bc27466bdff1178cea5a9cec5ed7",
          "url": "https://github.com/equinor/ert/commit/f8ede6f539dec5f04be18ce0ec3835ebb5c6edd9"
        },
        "date": 1709818507965,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1891673074510893,
            "unit": "iter/sec",
            "range": "stddev: 0.037157290806911734",
            "extra": "mean: 5.286325705400008 sec\nrounds: 5"
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
          "id": "43aa4dfa7bd61520e1033ca39cf9aec7155f907f",
          "message": "Do not retry when we get 'Job has finished' from qdel",
          "timestamp": "2024-03-07T15:05:28+01:00",
          "tree_id": "28af6ba416058e61b0d5d4d5394b83f2438e5487",
          "url": "https://github.com/equinor/ert/commit/43aa4dfa7bd61520e1033ca39cf9aec7155f907f"
        },
        "date": 1709820509411,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19260798518759042,
            "unit": "iter/sec",
            "range": "stddev: 0.17481567407492699",
            "extra": "mean: 5.191892740200001 sec\nrounds: 5"
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
          "id": "dd13514d59d1920742a9180cb88109740a8d5d6c",
          "message": "Add test of storage migration",
          "timestamp": "2024-03-07T15:31:31+01:00",
          "tree_id": "de88cb05d4ee5a792e66969e50fb3d4caf638b72",
          "url": "https://github.com/equinor/ert/commit/dd13514d59d1920742a9180cb88109740a8d5d6c"
        },
        "date": 1709822093350,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18760924165095055,
            "unit": "iter/sec",
            "range": "stddev: 0.08989905272022154",
            "extra": "mean: 5.33022782459999 sec\nrounds: 5"
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
          "id": "d33a48d88a22e58297cb1aeffe17d9aecb44e453",
          "message": "Fix calculation of batch size\n\n* Fix calculation of batch size\r\n\r\nNeed to take into account number of parameters.\r\n\r\n* Use float32 when calculating batch size",
          "timestamp": "2024-03-07T16:42:14+01:00",
          "tree_id": "3e25339bd43402d320c60ae6c9988e402ab91841",
          "url": "https://github.com/equinor/ert/commit/d33a48d88a22e58297cb1aeffe17d9aecb44e453"
        },
        "date": 1709826361504,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1437628396529341,
            "unit": "iter/sec",
            "range": "stddev: 0.13529948559944052",
            "extra": "mean: 6.955900442799793 sec\nrounds: 5"
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
          "id": "6d260a0aa3ef5503e310acc8f94da799edd31535",
          "message": "Dodge bug when simulation_context tears down (#7354)\n\nThis dodges a bug in either simulation_context or in scheduler.py that\nis triggered by test_stop_sim(). This checks the behaviour when all\nrealizations are killed, and there is a bug that can cause this process\nto hang while trying to kill.\n\nThis commit will rerun the test in case of failure, and will make the CI\nwork while the underlying bug is being prioritized.",
          "timestamp": "2024-03-07T16:44:37+01:00",
          "tree_id": "7d281ebb35ebc777553698b6a55ac724eff92638",
          "url": "https://github.com/equinor/ert/commit/6d260a0aa3ef5503e310acc8f94da799edd31535"
        },
        "date": 1709826468327,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1412421837156037,
            "unit": "iter/sec",
            "range": "stddev: 0.03344512360659894",
            "extra": "mean: 7.0800378024 sec\nrounds: 5"
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
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "da3f122acfae4a7274a787c71e844c4e3d74d974",
          "message": "Upgrade submodule",
          "timestamp": "2024-03-07T17:24:33+01:00",
          "tree_id": "8e6730bfb2495bdb5f595e60e0972e9181d961c6",
          "url": "https://github.com/equinor/ert/commit/da3f122acfae4a7274a787c71e844c4e3d74d974"
        },
        "date": 1709828886127,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.13564243031652687,
            "unit": "iter/sec",
            "range": "stddev: 0.05020327753452046",
            "extra": "mean: 7.372324409600014 sec\nrounds: 5"
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
          "id": "b1fc33adcfdb5df473ab6eab302e562846b2d578",
          "message": "Remove -s from pytest in CI\n\n`pytest-xdist` is incompatible with `-s`, and the output from successful\ntests isn't interesting.\n\nRef: https://github.com/pytest-dev/pytest-xdist/blob/017cc72b7090dc4bb7e7ad3d0caab024feb977a8/docs/known-limitations.rst#output-stdout-and-stderr-from-workers",
          "timestamp": "2024-03-07T20:22:38+01:00",
          "tree_id": "94447fb9b6b1a85e74411f9ef220d478b913f8f7",
          "url": "https://github.com/equinor/ert/commit/b1fc33adcfdb5df473ab6eab302e562846b2d578"
        },
        "date": 1709839561567,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.14140946083526335,
            "unit": "iter/sec",
            "range": "stddev: 0.039355385383929506",
            "extra": "mean: 7.071662632000004 sec\nrounds: 5"
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
          "id": "5b3f0702f87d32955a58cc0b7991a55afdce582a",
          "message": "Avoid \"qstat -f\" for non-finished jobs in PBS (#7376)\n\nqstat -f (f for \"full format\") is a heavy operation for the PBS cluster,\r\nand we only use it to obtain the Exit status for the job.\r\n\r\nThis will change the polling into using qstat -f only for jobs\r\nthat are already marked as finished, through polling with\r\nthe default qstat output.",
          "timestamp": "2024-03-08T06:42:37Z",
          "tree_id": "d71af2ef8783b09356126838b9ec28513c77ec32",
          "url": "https://github.com/equinor/ert/commit/5b3f0702f87d32955a58cc0b7991a55afdce582a"
        },
        "date": 1709880353352,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1405057775419177,
            "unit": "iter/sec",
            "range": "stddev: 0.03569247558360399",
            "extra": "mean: 7.117145056200025 sec\nrounds: 5"
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
          "id": "d015dfa8997a21dd385e5267b4b0908f2fcd3631",
          "message": "Revert calculation of batch size fix\n\nReverts this commit but adds derivation of formula to not make the same mistake again:\r\nd33a48d88a22e58297cb1aeffe17d9aecb44e453",
          "timestamp": "2024-03-08T09:19:24+01:00",
          "tree_id": "8b30f9169faec354b136bd944d651f8dc02b1478",
          "url": "https://github.com/equinor/ert/commit/d015dfa8997a21dd385e5267b4b0908f2fcd3631"
        },
        "date": 1709886175993,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1904727588899406,
            "unit": "iter/sec",
            "range": "stddev: 0.05122275832953483",
            "extra": "mean: 5.250094584800036 sec\nrounds: 5"
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
          "id": "bd37ca9a32629a1fa67f00dd47a1817e7b20081c",
          "message": "Revert \"Rename Field to FieldConfig\"\n\nThis reverts commit 119c8731cbd2677f10ba1aa0be8502603eb9b6c5.",
          "timestamp": "2024-03-08T09:45:08+01:00",
          "tree_id": "b4f0e6eca6d806ec731adda588a8aff42ed9a276",
          "url": "https://github.com/equinor/ert/commit/bd37ca9a32629a1fa67f00dd47a1817e7b20081c"
        },
        "date": 1709887693376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1912317401900545,
            "unit": "iter/sec",
            "range": "stddev: 0.05409308758257642",
            "extra": "mean: 5.229257439199978 sec\nrounds: 5"
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
          "id": "006499cff7d98620446c45872d8d92985dd512bb",
          "message": "Ensure no duplicate keys in SummaryConfig",
          "timestamp": "2024-03-08T12:25:52+01:00",
          "tree_id": "0365d3aa47e52a53eb16f5debb4a01ec8c1f12b8",
          "url": "https://github.com/equinor/ert/commit/006499cff7d98620446c45872d8d92985dd512bb"
        },
        "date": 1709897329769,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19094525803545884,
            "unit": "iter/sec",
            "range": "stddev: 0.02355417950002718",
            "extra": "mean: 5.237103085399997 sec\nrounds: 5"
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
          "id": "2f020324c51de074153dba7b6f7704073d1c366b",
          "message": "Add more cases to storage test",
          "timestamp": "2024-03-08T13:51:28+01:00",
          "tree_id": "1d2838ec14ea8c3cd619b4946762e572d46b6320",
          "url": "https://github.com/equinor/ert/commit/2f020324c51de074153dba7b6f7704073d1c366b"
        },
        "date": 1709902475605,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19223083853674058,
            "unit": "iter/sec",
            "range": "stddev: 0.03558162363451653",
            "extra": "mean: 5.202078956800017 sec\nrounds: 5"
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
          "id": "6d9656b344fe6feaa06b26b5d68f82a1e13c90dc",
          "message": "Treat exception in long lived tasks by cancelling the jobs correctly\n\nWe gather collectively results of realization tasks and scheduling_tasks\n(long live scheduling tasks). The main point is that the exceptions from realization tasks are treated differently than\nthe exceptions from scheduling tasks. Exception in scheduling tasks requires immidiate handling.\n\nThis includes unit tests when OpenPBS driver hanging / fails and\nscheduler related exception tests.\n\nAdditionally, this commit removes async_utils.background_tasks and test_async_utils.py\n\nCo-authored-by: Jonathan Karlsen <jonak@equinor.com>",
          "timestamp": "2024-03-10T16:45:41+01:00",
          "tree_id": "baea8d285c9011453884299ac96248b95e3a0f55",
          "url": "https://github.com/equinor/ert/commit/6d9656b344fe6feaa06b26b5d68f82a1e13c90dc"
        },
        "date": 1710085721837,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18745106016474103,
            "unit": "iter/sec",
            "range": "stddev: 0.08233757578141407",
            "extra": "mean: 5.334725763200015 sec\nrounds: 5"
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
          "id": "bbcb52bff3f2903a2bf55005e4992c2b0f628171",
          "message": "Fix external_ert_script does not fail on error (#7213)\n\n* Improve readability cli test test_that_stop_on_fail_workflow_jobs_stop_ert\r\n\r\n* Fix external_ert_script does not fail on error\r\n\r\n* Remove STOP_ON_FAIL keyword from scripts",
          "timestamp": "2024-03-11T10:15:23+01:00",
          "tree_id": "5b5fde9be6bcc61c02e3988d0a82c41b9a4a2ada",
          "url": "https://github.com/equinor/ert/commit/bbcb52bff3f2903a2bf55005e4992c2b0f628171"
        },
        "date": 1710148719294,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18888618221999529,
            "unit": "iter/sec",
            "range": "stddev: 0.08422267050500042",
            "extra": "mean: 5.294193509800005 sec\nrounds: 5"
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
          "id": "d09bd4e400db30d8ad612877b01016dfdd7f7c4b",
          "message": "Use newer style v resource allocation for Torque (C) and PBS (Python) (#7389)\n\nUse newer style resource allocation for qsub\r\n\r\nIn short; nodes replaced by select, and ppn replaced by ncpus.",
          "timestamp": "2024-03-11T12:45:22+01:00",
          "tree_id": "bd032e03c74c724b09c09b2f6e79ced0447f08fa",
          "url": "https://github.com/equinor/ert/commit/d09bd4e400db30d8ad612877b01016dfdd7f7c4b"
        },
        "date": 1710157712083,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1911255445449341,
            "unit": "iter/sec",
            "range": "stddev: 0.026319712460644306",
            "extra": "mean: 5.232162986800006 sec\nrounds: 5"
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
          "id": "6bac0ba8d54f6133c8017ca0ed55f7b2670c2084",
          "message": "Use a compiled regex for matching in summary\n\nlooping over a fnmatch was simply too slow",
          "timestamp": "2024-03-11T15:03:45+01:00",
          "tree_id": "0615b32af143c921ed2e7e2b5af6955462df06b8",
          "url": "https://github.com/equinor/ert/commit/6bac0ba8d54f6133c8017ca0ed55f7b2670c2084"
        },
        "date": 1710166011955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19082239886385385,
            "unit": "iter/sec",
            "range": "stddev: 0.020303577191101322",
            "extra": "mean: 5.240474943999999 sec\nrounds: 5"
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
          "id": "802375058dbcc08709945c84088133b7375ac935",
          "message": "Use block storage path fixture in storage test",
          "timestamp": "2024-03-12T08:39:36+01:00",
          "tree_id": "6ece5e6fdc73f8ad0b7aafa60807300f217d57ed",
          "url": "https://github.com/equinor/ert/commit/802375058dbcc08709945c84088133b7375ac935"
        },
        "date": 1710229354182,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1918157586620159,
            "unit": "iter/sec",
            "range": "stddev: 0.024688740907438094",
            "extra": "mean: 5.2133360000000035 sec\nrounds: 5"
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
          "id": "e398b220b3206bcb73d87540446faf24fc7286f9",
          "message": "Improve scheduler unit tests for scheduling_tasks exceptions (#7424)\n\nThis commit makes the unit tests more realistic by having specific error prone parts of the code raise exceptions rather than the entire scheduler method.",
          "timestamp": "2024-03-12T13:25:20+01:00",
          "tree_id": "8d75019569ce96390b54ebcee6fee83e47a6ce3f",
          "url": "https://github.com/equinor/ert/commit/e398b220b3206bcb73d87540446faf24fc7286f9"
        },
        "date": 1710246534120,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19091866319180323,
            "unit": "iter/sec",
            "range": "stddev: 0.06961334348982877",
            "extra": "mean: 5.237832610399994 sec\nrounds: 5"
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
          "id": "55b9e3a3fc3b2f9c478a9f04c649c93bdfb1dd5d",
          "message": "Sort observations before returning\n\nThe order of the observations are important when doing updates\nas it influences the pertubations. Now this property is used for\nupdates, which means the order matters.",
          "timestamp": "2024-03-12T13:38:10+01:00",
          "tree_id": "bedfe4888901ed373e84bd0ac652a5f09317bc39",
          "url": "https://github.com/equinor/ert/commit/55b9e3a3fc3b2f9c478a9f04c649c93bdfb1dd5d"
        },
        "date": 1710247285783,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19144116702130767,
            "unit": "iter/sec",
            "range": "stddev: 0.027877887992406206",
            "extra": "mean: 5.223536899400005 sec\nrounds: 5"
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
          "id": "ebb1d3708a9c0a9d6d8517d0ff5ad7cb9a3ef6d1",
          "message": "Increase memory limit to 130MB in test_field_param_memory",
          "timestamp": "2024-03-12T14:47:56+01:00",
          "tree_id": "5b1cc41fbe62fc5c609edf23787d69d9c5ec138c",
          "url": "https://github.com/equinor/ert/commit/ebb1d3708a9c0a9d6d8517d0ff5ad7cb9a3ef6d1"
        },
        "date": 1710251475000,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19026543138485952,
            "unit": "iter/sec",
            "range": "stddev: 0.023928934932165753",
            "extra": "mean: 5.255815482199966 sec\nrounds: 5"
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
          "id": "2b180e98fe0f8ad1e6eda48daa6a9d9d87c6f0ec",
          "message": "Add remaining states to the working STATES for LSF driver\n\nThis commit moves status update into its own function\n(_process_job_udpate). Additionally, it adds two standalone\nstates FinishedJobFailure and FinishedJobSuccess.\nIt adds test for process_job_update that it handles all the states properly.",
          "timestamp": "2024-03-12T15:47:05+01:00",
          "tree_id": "2d528ce9027dbd93b7422b6c98f885cfd17e3e14",
          "url": "https://github.com/equinor/ert/commit/2b180e98fe0f8ad1e6eda48daa6a9d9d87c6f0ec"
        },
        "date": 1710255022244,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1916946984843449,
            "unit": "iter/sec",
            "range": "stddev: 0.051441809661722945",
            "extra": "mean: 5.216628357000007 sec\nrounds: 5"
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
          "id": "92ec0a84a1cd4acd8c154c584201c029868bf679",
          "message": "Update 'initialize from scratch' to only work on ensembles in UNDEFINED state",
          "timestamp": "2024-03-13T09:38:33+01:00",
          "tree_id": "4fff4a6cbb8493d2b6b8545af9cc9cbf36981b8b",
          "url": "https://github.com/equinor/ert/commit/92ec0a84a1cd4acd8c154c584201c029868bf679"
        },
        "date": 1710319293801,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18809528018316862,
            "unit": "iter/sec",
            "range": "stddev: 0.0712168067790596",
            "extra": "mean: 5.316454506600019 sec\nrounds: 5"
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
          "id": "d22053cf3ebd5c68a890dbdc4256df4881d25375",
          "message": "Show total time in update tab",
          "timestamp": "2024-03-13T09:59:56+01:00",
          "tree_id": "072e047752c118db36e7e6ac29b1413b56273dbc",
          "url": "https://github.com/equinor/ert/commit/d22053cf3ebd5c68a890dbdc4256df4881d25375"
        },
        "date": 1710320574011,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19222737731400147,
            "unit": "iter/sec",
            "range": "stddev: 0.016927445080984736",
            "extra": "mean: 5.202172624800005 sec\nrounds: 5"
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
          "id": "8717b4f24a9eb5d1a5bd3af8a92123ac62360792",
          "message": "Rename simulation arguments",
          "timestamp": "2024-03-13T10:00:38+01:00",
          "tree_id": "2e1e62cf25303f483e5871409ffce002f214bb0f",
          "url": "https://github.com/equinor/ert/commit/8717b4f24a9eb5d1a5bd3af8a92123ac62360792"
        },
        "date": 1710320641892,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1857615228521758,
            "unit": "iter/sec",
            "range": "stddev: 0.09841114640663373",
            "extra": "mean: 5.383246135400031 sec\nrounds: 5"
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
          "id": "5e5ed6f8c20ff4294eaf1e9f731a88046eb922f6",
          "message": "Add more info about observations",
          "timestamp": "2024-03-13T10:25:50+01:00",
          "tree_id": "4c58745ddcf15d36b06dee39c514af308dbc5624",
          "url": "https://github.com/equinor/ert/commit/5e5ed6f8c20ff4294eaf1e9f731a88046eb922f6"
        },
        "date": 1710322162848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18997134500651605,
            "unit": "iter/sec",
            "range": "stddev: 0.03260841325770118",
            "extra": "mean: 5.263951781600008 sec\nrounds: 5"
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
          "id": "e65d1680b17055c81626e6a5a7c870d6aae59045",
          "message": "Parameterize test",
          "timestamp": "2024-03-13T12:20:06+01:00",
          "tree_id": "6fc07e0b15a750cb0d6934ca30660212951ea7eb",
          "url": "https://github.com/equinor/ert/commit/e65d1680b17055c81626e6a5a7c870d6aae59045"
        },
        "date": 1710328986534,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1870301541853596,
            "unit": "iter/sec",
            "range": "stddev: 0.13456434456725036",
            "extra": "mean: 5.346731409999973 sec\nrounds: 5"
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
          "id": "3381e679a4a9b5c436dd3c4701ffbd565301db0b",
          "message": "Deprecate unused lsf options (#7428)\n\n* Add deprecation warnings for outdated LSF queue options\r\n\r\nThis commit adds deprecation warnings for the outdated LSF queue options: LSF_LOGIN_SHELL, LSF_RSH_CMD.\r\n\r\n* Fix config schema not handling duplicate keyword deprecations\r\n\r\nThis commit makes it so that one keyword (for example QUEUE_CONFIG) can\r\nhave multiple deprecation warnings.\r\n\r\n* Remove deprecated LSF queue options from docs\r\n\r\nThis commit removed the deprecated LSF queue options: LSF_LOGIN_SHELL and LSF_RSH_CMD from the docs.",
          "timestamp": "2024-03-13T13:30:32+01:00",
          "tree_id": "559f4784f614a0109c44b1ec233792fc07a5b82d",
          "url": "https://github.com/equinor/ert/commit/3381e679a4a9b5c436dd3c4701ffbd565301db0b"
        },
        "date": 1710333212792,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1896308087239947,
            "unit": "iter/sec",
            "range": "stddev: 0.042357940057563125",
            "extra": "mean: 5.273404710599993 sec\nrounds: 5"
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
          "id": "2a451b8da610fe32df0f62f213fe2dcf8cca43dd",
          "message": "Add migration for ert_kind",
          "timestamp": "2024-03-13T14:34:48+01:00",
          "tree_id": "443d49677be21509828d4829305dfecc6d0163d6",
          "url": "https://github.com/equinor/ert/commit/2a451b8da610fe32df0f62f213fe2dcf8cca43dd"
        },
        "date": 1710337072444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1885534016903004,
            "unit": "iter/sec",
            "range": "stddev: 0.02173812389661609",
            "extra": "mean: 5.303537305799995 sec\nrounds: 5"
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
          "id": "eee4d2ac2e2b0a26c08d84208e6f794d5b89d51b",
          "message": "Include LocalDriver in generic driver tests",
          "timestamp": "2024-03-13T15:25:09+01:00",
          "tree_id": "967054f6b3d650c2708cf1974ecb6e3989edc43b",
          "url": "https://github.com/equinor/ert/commit/eee4d2ac2e2b0a26c08d84208e6f794d5b89d51b"
        },
        "date": 1710340088860,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19127202184314754,
            "unit": "iter/sec",
            "range": "stddev: 0.08026117134082517",
            "extra": "mean: 5.228156164000029 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}