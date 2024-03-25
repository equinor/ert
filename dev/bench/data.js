window.BENCHMARK_DATA = {
  "lastUpdate": 1711374657306,
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
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "01edea6b5a19132dc15a258670e23047ca949c76",
          "message": "Make PBS tests more tolerant for SIGTERM variants\n\nAt least on RHEL8 and Python 3.11 the return code from PBS has been observed to\nbe 128 + SIGTERM while at least on RHEL7/Python 3.8 it is 256 + SIGTERM.\n\nWe should accept both variants in tests.",
          "timestamp": "2024-03-15T07:29:08+01:00",
          "tree_id": "eb1e0ee320ce3d9e142ca166cd0ca6f7e58db089",
          "url": "https://github.com/equinor/ert/commit/01edea6b5a19132dc15a258670e23047ca949c76"
        },
        "date": 1710484340055,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18957470146407532,
            "unit": "iter/sec",
            "range": "stddev: 0.032078704209058714",
            "extra": "mean: 5.274965447800014 sec\nrounds: 5"
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
          "id": "26f4c2a86a6f4bd9c252c5149b407070a5a01893",
          "message": "Log failure to load ensemble",
          "timestamp": "2024-03-15T08:21:51+01:00",
          "tree_id": "2a56f6841b26148ff983c347fc8e71ecae4cfb5a",
          "url": "https://github.com/equinor/ert/commit/26f4c2a86a6f4bd9c252c5149b407070a5a01893"
        },
        "date": 1710487501100,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18850374373083603,
            "unit": "iter/sec",
            "range": "stddev: 0.030938167999382354",
            "extra": "mean: 5.304934428399986 sec\nrounds: 5"
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
          "id": "1e6d3fa50d7243289539ca1d10940e12adbf29eb",
          "message": "Add better feedback on no obs",
          "timestamp": "2024-03-15T08:59:23+01:00",
          "tree_id": "666b12318032b180003316c8005599e4637c0721",
          "url": "https://github.com/equinor/ert/commit/1e6d3fa50d7243289539ca1d10940e12adbf29eb"
        },
        "date": 1710489743219,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19250533091912939,
            "unit": "iter/sec",
            "range": "stddev: 0.021387836259504463",
            "extra": "mean: 5.194661338599997 sec\nrounds: 5"
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
          "id": "bfbbe09cb479904407c8e5c05ff4650cc6b2fdaa",
          "message": "Have OpenPBS driver use qstat -w option\n\n* Have OpenPBS driver use qstat -w option",
          "timestamp": "2024-03-15T09:10:55+01:00",
          "tree_id": "c660b1cd2e5e4a2541c239d95bb8fec68f88f3ca",
          "url": "https://github.com/equinor/ert/commit/bfbbe09cb479904407c8e5c05ff4650cc6b2fdaa"
        },
        "date": 1710490444897,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19060157249635876,
            "unit": "iter/sec",
            "range": "stddev: 0.033497533072988556",
            "extra": "mean: 5.246546431400003 sec\nrounds: 5"
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
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "ffae27e612339dc772ea7f9e6f81478b8da90262",
          "message": "Update contributing.md",
          "timestamp": "2024-03-15T09:22:11+01:00",
          "tree_id": "7d15e168eae4a3b32bbc397d95507207a0a77862",
          "url": "https://github.com/equinor/ert/commit/ffae27e612339dc772ea7f9e6f81478b8da90262"
        },
        "date": 1710491118891,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19040474037563954,
            "unit": "iter/sec",
            "range": "stddev: 0.03464584436605219",
            "extra": "mean: 5.2519700824000095 sec\nrounds: 5"
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
          "id": "fd5a365cd5c61e3ffe618d2d34b5da96d2aca23b",
          "message": "Rename `case` to `ensemble`\n\nReplace current_case and target_case with ensemble\r\n\r\nRemove requirement that ES run from CLI needs --target-case,\r\nbecause it is not required in the GUI.\r\n\r\nDeprecate current-case and target-case in CLI",
          "timestamp": "2024-03-15T11:40:08+01:00",
          "tree_id": "574b43dd34c3e445d13c0d82b8ef3d62c8893c01",
          "url": "https://github.com/equinor/ert/commit/fd5a365cd5c61e3ffe618d2d34b5da96d2aca23b"
        },
        "date": 1710499394825,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18783823355908133,
            "unit": "iter/sec",
            "range": "stddev: 0.129820003242732",
            "extra": "mean: 5.323729791599999 sec\nrounds: 5"
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
          "id": "a195daa86b002db5e83cf2a90ebdb5036a80c544",
          "message": "Fix LSF drivers ability to send to non-default queue\n\nThe Python LSF driver had a bug in a regexp causing it\nto crash if one tried to use a queue other than the default.",
          "timestamp": "2024-03-15T12:53:36+01:00",
          "tree_id": "3787e47856b7dbcd4b41e91c8cf41655425899c8",
          "url": "https://github.com/equinor/ert/commit/a195daa86b002db5e83cf2a90ebdb5036a80c544"
        },
        "date": 1710503799141,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18915112702360518,
            "unit": "iter/sec",
            "range": "stddev: 0.021877093235079602",
            "extra": "mean: 5.2867779100000005 sec\nrounds: 5"
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
          "id": "77e3139642d5b261764aa21f5aa4cf876e1bf7e7",
          "message": "Rewrite gui tests to have a clean app/storage for each test",
          "timestamp": "2024-03-15T12:57:12+01:00",
          "tree_id": "659c10c1988c04344e593c44319d71f3a03cbcbd",
          "url": "https://github.com/equinor/ert/commit/77e3139642d5b261764aa21f5aa4cf876e1bf7e7"
        },
        "date": 1710504025473,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1875578509976704,
            "unit": "iter/sec",
            "range": "stddev: 0.035833948220759214",
            "extra": "mean: 5.331688301399981 sec\nrounds: 5"
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
          "id": "cf6005e3eb9abf05a03380e8bfe2a8db7d70b340",
          "message": "Stop failing on upload error",
          "timestamp": "2024-03-15T13:54:57+01:00",
          "tree_id": "0a3029b87dc8d64f8265ee696fe0394ad611259d",
          "url": "https://github.com/equinor/ert/commit/cf6005e3eb9abf05a03380e8bfe2a8db7d70b340"
        },
        "date": 1710507496087,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19128430113086228,
            "unit": "iter/sec",
            "range": "stddev: 0.019113974030628315",
            "extra": "mean: 5.227820548200009 sec\nrounds: 5"
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
          "id": "819de9417a134f4fa6ba588e1d1c3b9d391b5166",
          "message": "Fix integration tests with real LSF cluster\n\nThe LSF driver writes its job script to disk and sends that script path\nto the LSF cluster through bsub. If the job script is not on a shared\ndisk the job will fail silently.",
          "timestamp": "2024-03-15T15:03:57+01:00",
          "tree_id": "ca5b095e7e69492d8300fb3268c6d6036846afc8",
          "url": "https://github.com/equinor/ert/commit/819de9417a134f4fa6ba588e1d1c3b9d391b5166"
        },
        "date": 1710511624623,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18992629412739429,
            "unit": "iter/sec",
            "range": "stddev: 0.04214575572585793",
            "extra": "mean: 5.265200400999999 sec\nrounds: 5"
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
          "id": "037606a9a8a5e3b09f41f0ef5b08da7c6b055471",
          "message": "Add handling for unknown module",
          "timestamp": "2024-03-15T20:02:37+01:00",
          "tree_id": "afd238f1a58aca2a1afda994b56634893303b6f4",
          "url": "https://github.com/equinor/ert/commit/037606a9a8a5e3b09f41f0ef5b08da7c6b055471"
        },
        "date": 1710529549733,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18415144575859044,
            "unit": "iter/sec",
            "range": "stddev: 0.03870528304095054",
            "extra": "mean: 5.430313055000011 sec\nrounds: 5"
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
          "id": "0d1b285e0b88c70770ffad820cec1ccf02fe10a0",
          "message": "Remove unused file exists exception",
          "timestamp": "2024-03-18T08:10:09+01:00",
          "tree_id": "fb1c0e762b9677d042de19b846d3a45e79fac5a4",
          "url": "https://github.com/equinor/ert/commit/0d1b285e0b88c70770ffad820cec1ccf02fe10a0"
        },
        "date": 1710746017134,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18929464354066275,
            "unit": "iter/sec",
            "range": "stddev: 0.02291182619699376",
            "extra": "mean: 5.282769661600002 sec\nrounds: 5"
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
          "id": "13e45707b2a232757ada36023dca904ee5d22c2e",
          "message": "Allow cleanup rm commands to fail\n\nThese rm commands operate on a NFS share, and removal on NFS\nshares occasionally fail due to some lock files or similar being\nhard to remove.",
          "timestamp": "2024-03-18T13:57:18+01:00",
          "tree_id": "fe82332448c34fa2783e93708553070aea89bf24",
          "url": "https://github.com/equinor/ert/commit/13e45707b2a232757ada36023dca904ee5d22c2e"
        },
        "date": 1710766903418,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1859576522602376,
            "unit": "iter/sec",
            "range": "stddev: 0.13081816100983812",
            "extra": "mean: 5.377568429400014 sec\nrounds: 5"
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
          "id": "e4d25a12f028144bc87a28906fd1f413c277d8ed",
          "message": "Ensure no overflow in parameter_example test",
          "timestamp": "2024-03-19T09:25:02+01:00",
          "tree_id": "1a2bbafcfd935959ab0d40ec619eee7fb41d75d1",
          "url": "https://github.com/equinor/ert/commit/e4d25a12f028144bc87a28906fd1f413c277d8ed"
        },
        "date": 1710836909982,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19106880284409453,
            "unit": "iter/sec",
            "range": "stddev: 0.0360058787994941",
            "extra": "mean: 5.233716782199997 sec\nrounds: 5"
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
          "id": "501709ce4a39f92ef0dc9973cbda81b11c625176",
          "message": "Add function for calculating std dev",
          "timestamp": "2024-03-19T12:45:51+01:00",
          "tree_id": "e266894d3976be62d03c5e78c8b2f0ade405ca28",
          "url": "https://github.com/equinor/ert/commit/501709ce4a39f92ef0dc9973cbda81b11c625176"
        },
        "date": 1710848933489,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19043773150013904,
            "unit": "iter/sec",
            "range": "stddev: 0.024980709164708355",
            "extra": "mean: 5.251060239599997 sec\nrounds: 5"
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
          "id": "30795c84c6d9634730a15d43e2605b29ac9ce995",
          "message": "Run memory tests seperately in testkomodo",
          "timestamp": "2024-03-19T13:52:21+01:00",
          "tree_id": "01c8d6ba24a0273c81fc5740b8b14be7c3d7f6d3",
          "url": "https://github.com/equinor/ert/commit/30795c84c6d9634730a15d43e2605b29ac9ce995"
        },
        "date": 1710852960917,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18676295136296484,
            "unit": "iter/sec",
            "range": "stddev: 0.04266729362238643",
            "extra": "mean: 5.354381009199988 sec\nrounds: 5"
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
          "id": "11f9b15a297e0dae84b96643b28ab8462566b1a6",
          "message": "Mark failing test test_openpbs_driver_with_poly_example_failing_poll_… (#7485)\n\nMark failing test test_openpbs_driver_with_poly_example_failing_poll_fails_ert_and_propagates_exception_to_user with xfail",
          "timestamp": "2024-03-20T12:00:15Z",
          "tree_id": "73bf42349049dd5bf91fb07a01591314be6556f5",
          "url": "https://github.com/equinor/ert/commit/11f9b15a297e0dae84b96643b28ab8462566b1a6"
        },
        "date": 1710936259542,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18297813247178285,
            "unit": "iter/sec",
            "range": "stddev: 0.1274411738542788",
            "extra": "mean: 5.465133928799991 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "JONAK@equinor.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "committer": {
            "email": "jparu@equinor.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "distinct": true,
          "id": "4423aad810e8df0d44bab57598211eab12414ff1",
          "message": "Have scheduler handle exceptions and log them as they arrive\n\nUse asyncio.wait to monitor tasks in order to get failures/exceptions immidiatally\nas they come. We just log the exceptions + the traceback coming from the job tasks,\nwhere in the end, we raise the first exception in realization tasks that was encountered.\nIf the exception occurs in scheduling_tasks we raise immidiatelly,\nwherein we firstly cancel all realization tasks.\n\nMoreover, when canceling the realizations tasks we exploit timeout to\ncancel the task in case that the realization be handing.\nThe standart self._tasks are renamed to self._job_tasks.\n\nCo-authored-by: Jonathan Karlsen <jonak@equinor.com>\nCo-authored-by: Julius Parulek <jparu@equinor.com>\n\nMOd.",
          "timestamp": "2024-03-20T15:26:43+01:00",
          "tree_id": "387964e8782b9eb3c70601744b2ff0ba19393e32",
          "url": "https://github.com/equinor/ert/commit/4423aad810e8df0d44bab57598211eab12414ff1"
        },
        "date": 1710945002601,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1883239042349834,
            "unit": "iter/sec",
            "range": "stddev: 0.02425383399232227",
            "extra": "mean: 5.310000363800009 sec\nrounds: 5"
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
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "111b2adfa09ebd731d8f8f69160a099f7e2b30c8",
          "message": "Skip flaky test_parameter_example",
          "timestamp": "2024-03-20T15:44:29+01:00",
          "tree_id": "b3d5c6424fb27c579aa5444ac335de0f6155bd88",
          "url": "https://github.com/equinor/ert/commit/111b2adfa09ebd731d8f8f69160a099f7e2b30c8"
        },
        "date": 1710946095592,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18584735829719595,
            "unit": "iter/sec",
            "range": "stddev: 0.0627480367220692",
            "extra": "mean: 5.3807598298000014 sec\nrounds: 5"
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
          "id": "6fa9a1367b967682fd53f3765ac9add1608801a5",
          "message": "Raise error on empty responses and parameters",
          "timestamp": "2024-03-20T17:56:07+01:00",
          "tree_id": "548d222a5c47241249602619c2347e4d55ad3b47",
          "url": "https://github.com/equinor/ert/commit/6fa9a1367b967682fd53f3765ac9add1608801a5"
        },
        "date": 1710953947796,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18749858681143391,
            "unit": "iter/sec",
            "range": "stddev: 0.043849147851290134",
            "extra": "mean: 5.333373530999961 sec\nrounds: 5"
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
          "id": "a71a6dfa90f622408c349eb6edf9e415d2d48f22",
          "message": "Have lsf_driver specify SIGKILL when bkilling (#7433)\n\nHave lsf_driver specify SIGKILL signal when using bkill",
          "timestamp": "2024-03-21T07:52:08+01:00",
          "tree_id": "791e4409e7dc2cad0bc1b9e255c56c64367f1e56",
          "url": "https://github.com/equinor/ert/commit/a71a6dfa90f622408c349eb6edf9e415d2d48f22"
        },
        "date": 1711004119044,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18753827906639295,
            "unit": "iter/sec",
            "range": "stddev: 0.01582365382085041",
            "extra": "mean: 5.332244728800015 sec\nrounds: 5"
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
          "id": "64fe249bc183f0d07df2dfee487c6d5a32b31e4e",
          "message": "Ensure correct array type with ES\n\nImportant to make sure fields that should be float32 never get\r\ncast to float64, which would waste memory.",
          "timestamp": "2024-03-21T09:39:26+01:00",
          "tree_id": "dff118769c6808a36b467b23c79acf8e0763225b",
          "url": "https://github.com/equinor/ert/commit/64fe249bc183f0d07df2dfee487c6d5a32b31e4e"
        },
        "date": 1711010563376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.188696566324733,
            "unit": "iter/sec",
            "range": "stddev: 0.05191145759745777",
            "extra": "mean: 5.299513496600002 sec\nrounds: 5"
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
          "id": "795982a5ef6658c0674368965aeb831f06cf9249",
          "message": "Unify returncode values for different drivers when process is killed by\nsignal\n\nThis should fix a bug in azure bleeding.",
          "timestamp": "2024-03-21T13:28:08+01:00",
          "tree_id": "938bc6c574589e2e1727abcf5fd2907adc07d47b",
          "url": "https://github.com/equinor/ert/commit/795982a5ef6658c0674368965aeb831f06cf9249"
        },
        "date": 1711024313021,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18739371171325114,
            "unit": "iter/sec",
            "range": "stddev: 0.02171807671367211",
            "extra": "mean: 5.336358359399992 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "JONAK@equinor.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "committer": {
            "email": "107626001+jonathan-eq@users.noreply.github.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "distinct": true,
          "id": "39dbee19267a55a48c8f6d2cd24370fad80f11d3",
          "message": "Add test for bjobs output with no exec_host",
          "timestamp": "2024-03-22T07:34:59+01:00",
          "tree_id": "6165561736be7f36454870ba1195c2b519b303d6",
          "url": "https://github.com/equinor/ert/commit/39dbee19267a55a48c8f6d2cd24370fad80f11d3"
        },
        "date": 1711089499396,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19032447707140412,
            "unit": "iter/sec",
            "range": "stddev: 0.03740357685161665",
            "extra": "mean: 5.2541849339999995 sec\nrounds: 5"
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
          "id": "ab421c85be0198656b80fcc7424d0d6f9d00bad7",
          "message": "Save storage version after migration",
          "timestamp": "2024-03-22T10:34:32+01:00",
          "tree_id": "840cd2ee0f70d6ab954a648cdcbb7fa2e391a6cb",
          "url": "https://github.com/equinor/ert/commit/ab421c85be0198656b80fcc7424d0d6f9d00bad7"
        },
        "date": 1711100258217,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18609876219132077,
            "unit": "iter/sec",
            "range": "stddev: 0.028961885165049814",
            "extra": "mean: 5.373490872399998 sec\nrounds: 5"
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
          "id": "692d2afc3ea8ab35e054c2aaa67f60117b93e52f",
          "message": "Fix single-test-run",
          "timestamp": "2024-03-22T10:45:31+01:00",
          "tree_id": "5e5fb7d860e1e6dab2ef31543a0d141c1bd7c232",
          "url": "https://github.com/equinor/ert/commit/692d2afc3ea8ab35e054c2aaa67f60117b93e52f"
        },
        "date": 1711100922194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19092129969581922,
            "unit": "iter/sec",
            "range": "stddev: 0.015281550932992907",
            "extra": "mean: 5.237760279199994 sec\nrounds: 5"
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
          "id": "fdf0c3c2e3e772745b6afc7fdce834a9ac822225",
          "message": "Remove unused keyword RESULT_PATH",
          "timestamp": "2024-03-22T10:47:39+01:00",
          "tree_id": "480e331e5fbf5fe76875fa516d2471b784af05b5",
          "url": "https://github.com/equinor/ert/commit/fdf0c3c2e3e772745b6afc7fdce834a9ac822225"
        },
        "date": 1711101049875,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19077649298930938,
            "unit": "iter/sec",
            "range": "stddev: 0.023799184829226314",
            "extra": "mean: 5.241735941000013 sec\nrounds: 5"
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
          "id": "5d32021da096a038a5d2fae5095a99190bb5a189",
          "message": "Add a test for opening empty storage",
          "timestamp": "2024-03-22T12:04:43+01:00",
          "tree_id": "64e693c55a241a92a495a13ac95696af95445ccc",
          "url": "https://github.com/equinor/ert/commit/5d32021da096a038a5d2fae5095a99190bb5a189"
        },
        "date": 1711105665606,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1854027183388136,
            "unit": "iter/sec",
            "range": "stddev: 0.045628335491161395",
            "extra": "mean: 5.3936641758000174 sec\nrounds: 5"
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
          "id": "05997ebe7341781bc2b6a6d5de9c5a59cab4290f",
          "message": "Fix If -> if in docs\n\nBonus: Reformat the paragraph for linelengths",
          "timestamp": "2024-03-22T14:31:11+01:00",
          "tree_id": "f3b21e28abd62e9f6edb67782b9b38ec2113b9f4",
          "url": "https://github.com/equinor/ert/commit/05997ebe7341781bc2b6a6d5de9c5a59cab4290f"
        },
        "date": 1711114459872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1891019844104315,
            "unit": "iter/sec",
            "range": "stddev: 0.06482650703391217",
            "extra": "mean: 5.288151803999983 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "JONAK@equinor.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "committer": {
            "email": "107626001+jonathan-eq@users.noreply.github.com",
            "name": "Jonathan Karlsen",
            "username": "jonathan-eq"
          },
          "distinct": true,
          "id": "42c2fd2557079392817c10e61f2f4c8651ee7bfd",
          "message": "Increase sleep time flaky integration test\n\nThis commit fixes flakiness in integration test scheduler/test_generic_driver.py::test_kill by increasing sleep time for job to 60 seconds. It might have been flaky in the past due to job finishing before it could be killed.",
          "timestamp": "2024-03-25T14:47:46+01:00",
          "tree_id": "5f1e7875d5eda40d9c2c91ba0f59b08f600fb689",
          "url": "https://github.com/equinor/ert/commit/42c2fd2557079392817c10e61f2f4c8651ee7bfd"
        },
        "date": 1711374656710,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18301420180832076,
            "unit": "iter/sec",
            "range": "stddev: 0.15123690416542848",
            "extra": "mean: 5.464056833399991 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}