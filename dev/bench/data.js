window.BENCHMARK_DATA = {
  "lastUpdate": 1736237936982,
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
          "id": "1bd4fe1294258050dd495e9a33ba3d23f47149b7",
          "message": "Remove unused async_run",
          "timestamp": "2024-12-20T14:06:05+01:00",
          "tree_id": "85bf2c979daf0ca04f6fd2a58006b95445cbe925",
          "url": "https://github.com/equinor/ert/commit/1bd4fe1294258050dd495e9a33ba3d23f47149b7"
        },
        "date": 1734700072391,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21569730252161742,
            "unit": "iter/sec",
            "range": "stddev: 0.03573023895218398",
            "extra": "mean: 4.636126591800002 sec\nrounds: 5"
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
          "id": "8fa3040800cc5253dc1ed5a34c8ceb32281373c7",
          "message": "Remove outdated logging handle",
          "timestamp": "2024-12-20T14:07:24+01:00",
          "tree_id": "0d92b4c9a2b2f21d743ac9e00697aa34bc0a79df",
          "url": "https://github.com/equinor/ert/commit/8fa3040800cc5253dc1ed5a34c8ceb32281373c7"
        },
        "date": 1734700155622,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21838175275939375,
            "unit": "iter/sec",
            "range": "stddev: 0.03988909457111767",
            "extra": "mean: 4.579137164000002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "committer": {
            "email": "pieter.verveer@tno.nl",
            "name": "Peter Verveer",
            "username": "verveerpj"
          },
          "distinct": true,
          "id": "722e0fc805cde27add2a3ad3344fcfc99fdf5b10",
          "message": "EverestRunModel: sanitize realization var names",
          "timestamp": "2024-12-20T14:27:56+01:00",
          "tree_id": "ed41627d26d55693d518eb54f9bad50b37092805",
          "url": "https://github.com/equinor/ert/commit/722e0fc805cde27add2a3ad3344fcfc99fdf5b10"
        },
        "date": 1734701386950,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21948512170606432,
            "unit": "iter/sec",
            "range": "stddev: 0.019461140551392343",
            "extra": "mean: 4.5561174817999985 sec\nrounds: 5"
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
          "id": "9b1a01e384a8ee51ba79179633afa44f85f9d070",
          "message": "Implementing router-dealer pattern with custom acknowledgments with zmq\n\n - dealers always wait for acknowledgment from the evaluator\n - removing websockets and wait_for_evaluator\n - Settup encryption with curve\n - each dealer (client, dispatcher) will get a unique name\n - Monitor is an advanced version Client\n - _server_started.wait() is to signal that zmq router socket is bound\n - Use TCP protocol only when using LSF, SLURM or TORQUE queues\n -- Use ipc_protocol when using LOCAL driver\n - Remove certificate\n - Remove synced _send from Client\n - Remove cert generator\n - Remove ClientConnectionClosedOK\n - Add test for new connection while closing down evaluator\n - Add test for handle dispatcher and dispatcher messages in evaluator\n - Add tests for ipc and tcp ee config\n - Add test for clear connect and disconnect of Monitor\n - Set a a correct protocol for everestserver",
          "timestamp": "2024-12-20T15:10:48+01:00",
          "tree_id": "97f8232e6ba8bbfc1633b74ddd045360372e50f5",
          "url": "https://github.com/equinor/ert/commit/9b1a01e384a8ee51ba79179633afa44f85f9d070"
        },
        "date": 1734703955491,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2213877592052432,
            "unit": "iter/sec",
            "range": "stddev: 0.03837322182960567",
            "extra": "mean: 4.516961568200003 sec\nrounds: 5"
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
          "id": "b13ad454dbeceaee6a3f3470c9d95693d0ca9a02",
          "message": "Remove deprecated torque options\n\nThis commit removes the deprecated torque/openpbs queue options:\n* QUEUE_QUERY_TIMEOUT\n* NUM_NODES\n* NUM_CPUS_PER_NODE\n* QSTAT_OPTIONS\n* MEMORY_PER_JOB",
          "timestamp": "2024-12-20T15:29:08+01:00",
          "tree_id": "8bb6563900111aa9740d7bf5d3083038ff475345",
          "url": "https://github.com/equinor/ert/commit/b13ad454dbeceaee6a3f3470c9d95693d0ca9a02"
        },
        "date": 1734705055699,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21959705996330614,
            "unit": "iter/sec",
            "range": "stddev: 0.023333546213795374",
            "extra": "mean: 4.553795028800005 sec\nrounds: 5"
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
          "id": "855c517890b2342f82907a11ae519aec2b50d885",
          "message": "Remove unused type ignore.",
          "timestamp": "2024-12-23T18:03:09+02:00",
          "tree_id": "78a1ee71b253db603121844eea9e740b147e1662",
          "url": "https://github.com/equinor/ert/commit/855c517890b2342f82907a11ae519aec2b50d885"
        },
        "date": 1734969896128,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21858073960126662,
            "unit": "iter/sec",
            "range": "stddev: 0.008085150111211182",
            "extra": "mean: 4.574968507400024 sec\nrounds: 5"
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
          "id": "9ba6c47784c8c7ad59a84ea6a645f42c759721eb",
          "message": "Avoid warning from pytest\n\nPytestWarning: Value of environment variable ERT_STORAGE_ENS_PATH type\nshould be str, but got PosixPath('/.../storage') (type: PosixPath);\nconverted to str implicitly",
          "timestamp": "2025-01-02T09:05:09+01:00",
          "tree_id": "f12f6a458868d0303e7fc7d5524ade45dd29d3b8",
          "url": "https://github.com/equinor/ert/commit/9ba6c47784c8c7ad59a84ea6a645f42c759721eb"
        },
        "date": 1735805219779,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22087436811815722,
            "unit": "iter/sec",
            "range": "stddev: 0.015636184030795086",
            "extra": "mean: 4.527460603599996 sec\nrounds: 5"
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
          "id": "929d59b73cea011aafc95bd62a950d963928cf64",
          "message": "Avoid using deprecated --target-case in tests",
          "timestamp": "2025-01-02T09:05:27+01:00",
          "tree_id": "b6dc1b1cae0e7a3c6f0c6767c52da7b975f55dd5",
          "url": "https://github.com/equinor/ert/commit/929d59b73cea011aafc95bd62a950d963928cf64"
        },
        "date": 1735805240505,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2130313133209388,
            "unit": "iter/sec",
            "range": "stddev: 0.01825512749633529",
            "extra": "mean: 4.694145590200003 sec\nrounds: 5"
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
          "id": "45a4b7ca72b6bdda0a777d7a4712d19d97ab32b6",
          "message": "Workaround Python bug for shutil.copytree() exceptions\n\nSee https://github.com/python/cpython/issues/102931\n\nThis PR will detect if the bug is triggered, and massage the data\naccordingly. Patch is prepared upstream destined for Python 3.14.\n\nExisting tests are split for readability and preciseness, and extended\nto test that the workaround is performing as it should.",
          "timestamp": "2025-01-02T09:50:23+01:00",
          "tree_id": "c547f6e90012d98758f3c8a6061b15f77febb069",
          "url": "https://github.com/equinor/ert/commit/45a4b7ca72b6bdda0a777d7a4712d19d97ab32b6"
        },
        "date": 1735807930663,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2204107451657714,
            "unit": "iter/sec",
            "range": "stddev: 0.0216063470528396",
            "extra": "mean: 4.536983889999999 sec\nrounds: 5"
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
          "id": "9e2e32f88af2a4eec85cacdaddc47ba4af2de227",
          "message": "Avoid deprecated utcnow()",
          "timestamp": "2025-01-02T12:33:28+01:00",
          "tree_id": "46ad2b0faf1c940fa80a69961197d31097d0cbb9",
          "url": "https://github.com/equinor/ert/commit/9e2e32f88af2a4eec85cacdaddc47ba4af2de227"
        },
        "date": 1735817713031,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21773301885494384,
            "unit": "iter/sec",
            "range": "stddev: 0.03724143083640846",
            "extra": "mean: 4.592780669000007 sec\nrounds: 5"
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
          "id": "135c3c9053c91364bc957d979c42a8365c124c6a",
          "message": "Avoid repeated logging for each cluster in misfit analysis",
          "timestamp": "2025-01-02T15:48:01+01:00",
          "tree_id": "f1ed2b06ce0701e3b0e12ccf96829f5a95274d16",
          "url": "https://github.com/equinor/ert/commit/135c3c9053c91364bc957d979c42a8365c124c6a"
        },
        "date": 1735829396671,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21640452699990395,
            "unit": "iter/sec",
            "range": "stddev: 0.018154236135735673",
            "extra": "mean: 4.620975419799993 sec\nrounds: 5"
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
          "id": "7e8d20cdc4001980021b6b51b87e76e53a5c6dfc",
          "message": "Change needed for updated everest-models",
          "timestamp": "2025-01-03T09:23:55+01:00",
          "tree_id": "68e380011fb2204e36e75a0dd9ff56f48f40d243",
          "url": "https://github.com/equinor/ert/commit/7e8d20cdc4001980021b6b51b87e76e53a5c6dfc"
        },
        "date": 1735892743897,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22004640687181604,
            "unit": "iter/sec",
            "range": "stddev: 0.023821548294085563",
            "extra": "mean: 4.5444959280000035 sec\nrounds: 5"
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
          "id": "a372383de1d69a2b0da463f1fd6897e5cd50b3e0",
          "message": "Pin iterative_ensemble_smoother\n\nTests are not working with 0.3.0",
          "timestamp": "2025-01-03T13:11:29+01:00",
          "tree_id": "8a9d3cc3d4034939aa0154276fc62cf534a14b81",
          "url": "https://github.com/equinor/ert/commit/a372383de1d69a2b0da463f1fd6897e5cd50b3e0"
        },
        "date": 1735906397422,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21740204916053793,
            "unit": "iter/sec",
            "range": "stddev: 0.03137568113211991",
            "extra": "mean: 4.599772651 sec\nrounds: 5"
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
          "id": "987bb0b555b5f77d07ba64c7a90433adba9eaca2",
          "message": "Add property test of ensemble_smoother",
          "timestamp": "2025-01-03T13:31:39+01:00",
          "tree_id": "796aa000b707084c8f003344ebeaf67393e2d9f9",
          "url": "https://github.com/equinor/ert/commit/987bb0b555b5f77d07ba64c7a90433adba9eaca2"
        },
        "date": 1735907605571,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21585071692996452,
            "unit": "iter/sec",
            "range": "stddev: 0.049524646250869074",
            "extra": "mean: 4.632831496799997 sec\nrounds: 5"
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
          "id": "6a823997185e298d1eed3266c1615a7beb2158a1",
          "message": "Use new version of iterative_ensemble_smoother\n\nReturns all correlated pairs and not just those deemed\nsignificantly correlated.\nTherefore, some tests need to be updated.",
          "timestamp": "2025-01-03T13:45:12+01:00",
          "tree_id": "3795a98a8a1b82cb741a9ce056d730729908f5b7",
          "url": "https://github.com/equinor/ert/commit/6a823997185e298d1eed3266c1615a7beb2158a1"
        },
        "date": 1735908422013,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22117051425708484,
            "unit": "iter/sec",
            "range": "stddev: 0.036435514528603996",
            "extra": "mean: 4.521398358000004 sec\nrounds: 5"
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
          "id": "719a81e8b66424dddacfc4564255d0ff1fb9a361",
          "message": "Set up dependabot for github actions",
          "timestamp": "2025-01-06T08:22:45+01:00",
          "tree_id": "3a6bf743cedc1801f8e1c0c4ae703f959d75acf9",
          "url": "https://github.com/equinor/ert/commit/719a81e8b66424dddacfc4564255d0ff1fb9a361"
        },
        "date": 1736148274354,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.20463628202251075,
            "unit": "iter/sec",
            "range": "stddev: 0.0255482766256843",
            "extra": "mean: 4.886718963599995 sec\nrounds: 5"
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
          "id": "864c1e85cbd8f1ce8f66d26a915cbe1f6791245f",
          "message": "Pin scipy < 1.15",
          "timestamp": "2025-01-06T10:25:29+01:00",
          "tree_id": "4a6e7a94b057f606528c49cdb00b9e856154a150",
          "url": "https://github.com/equinor/ert/commit/864c1e85cbd8f1ce8f66d26a915cbe1f6791245f"
        },
        "date": 1736155642188,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21938858886307655,
            "unit": "iter/sec",
            "range": "stddev: 0.04083733083937065",
            "extra": "mean: 4.558122212199987 sec\nrounds: 5"
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
          "id": "839610bb7c6a249cf03fdccf7f0e467b68a9f88a",
          "message": "Ignore unimportant ConfigWarnings",
          "timestamp": "2025-01-06T11:30:00+01:00",
          "tree_id": "253d8cc5d9ef311f8e13a8010763c96abdc83c60",
          "url": "https://github.com/equinor/ert/commit/839610bb7c6a249cf03fdccf7f0e467b68a9f88a"
        },
        "date": 1736159518362,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21533315528316188,
            "unit": "iter/sec",
            "range": "stddev: 0.043433576491164506",
            "extra": "mean: 4.643966688199993 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "60844986+larsevj@users.noreply.github.com",
            "name": "Lars Evje",
            "username": "larsevj"
          },
          "distinct": true,
          "id": "7d73f740d8f12d2a0a726782fb7753acd3c06ef2",
          "message": "Bump astral-sh/setup-uv from 4 to 5\n\nBumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 4 to 5.\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/v4...v5)\n\n---\nupdated-dependencies:\n- dependency-name: astral-sh/setup-uv\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2025-01-06T12:33:59+01:00",
          "tree_id": "720473b4719f3cd6476a3d40c89d3d3ed5ae2b85",
          "url": "https://github.com/equinor/ert/commit/7d73f740d8f12d2a0a726782fb7753acd3c06ef2"
        },
        "date": 1736163347409,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21680333050116213,
            "unit": "iter/sec",
            "range": "stddev: 0.026559978642120226",
            "extra": "mean: 4.612475268199995 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "60844986+larsevj@users.noreply.github.com",
            "name": "Lars Evje",
            "username": "larsevj"
          },
          "distinct": true,
          "id": "ab373dd5930aec8fe06622df2e0bd8100b0a750b",
          "message": "Bump pypa/gh-action-pypi-publish from 1.10.1 to 1.12.3\n\nBumps [pypa/gh-action-pypi-publish](https://github.com/pypa/gh-action-pypi-publish) from 1.10.1 to 1.12.3.\n- [Release notes](https://github.com/pypa/gh-action-pypi-publish/releases)\n- [Commits](https://github.com/pypa/gh-action-pypi-publish/compare/v1.10.1...v1.12.3)\n\n---\nupdated-dependencies:\n- dependency-name: pypa/gh-action-pypi-publish\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2025-01-06T12:36:59+01:00",
          "tree_id": "7be64911980eac6e06a16265f0222028a84c63d8",
          "url": "https://github.com/equinor/ert/commit/ab373dd5930aec8fe06622df2e0bd8100b0a750b"
        },
        "date": 1736163525935,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2162420984924752,
            "unit": "iter/sec",
            "range": "stddev: 0.029626205247944",
            "extra": "mean: 4.624446428200002 sec\nrounds: 5"
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
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "68263ff9a8e4675e59a6a98b43bdc765e11cf1a0",
          "message": "Fixed plotter storing correct tab when switching plot types",
          "timestamp": "2025-01-06T12:58:53+01:00",
          "tree_id": "0ccef630c0b333979236f1c2a9f3e303c59babac",
          "url": "https://github.com/equinor/ert/commit/68263ff9a8e4675e59a6a98b43bdc765e11cf1a0"
        },
        "date": 1736164846339,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2146418837051802,
            "unit": "iter/sec",
            "range": "stddev: 0.05606033087355752",
            "extra": "mean: 4.658922959199998 sec\nrounds: 5"
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
          "id": "3cee24dc05262c38bc9116d9228f7d770401306b",
          "message": "Validate derrf distribution parameters on startup",
          "timestamp": "2025-01-06T13:30:18+01:00",
          "tree_id": "1446c51c808304c04e32f0fba8168a458c07fa06",
          "url": "https://github.com/equinor/ert/commit/3cee24dc05262c38bc9116d9228f7d770401306b"
        },
        "date": 1736166725834,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2168873548856917,
            "unit": "iter/sec",
            "range": "stddev: 0.08776663116942524",
            "extra": "mean: 4.610688348000002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ynk@equinor.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "committer": {
            "email": "yngve-sk@users.noreply.github.com",
            "name": "Yngve S. Kristiansen",
            "username": "yngve-sk"
          },
          "distinct": true,
          "id": "fe0c40a80293b4e008f002d3f54cfbeb06eb10c6",
          "message": "Use cached_example for test_everest_entry",
          "timestamp": "2025-01-06T13:39:54+01:00",
          "tree_id": "d612888bea57df022e22600083e51f2bf2bb5739",
          "url": "https://github.com/equinor/ert/commit/fe0c40a80293b4e008f002d3f54cfbeb06eb10c6"
        },
        "date": 1736167304404,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2112923232972261,
            "unit": "iter/sec",
            "range": "stddev: 0.06383405514701589",
            "extra": "mean: 4.732779612600001 sec\nrounds: 5"
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
          "id": "18fb2b1a903c7868a48dd675e78b5396b0d4c979",
          "message": "Split asserts combined with and for better error message",
          "timestamp": "2025-01-06T13:54:51+01:00",
          "tree_id": "540ec38484d0e9a0882c9ad5abd19d5d7bdab96c",
          "url": "https://github.com/equinor/ert/commit/18fb2b1a903c7868a48dd675e78b5396b0d4c979"
        },
        "date": 1736168201192,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21228887608459737,
            "unit": "iter/sec",
            "range": "stddev: 0.11823279806557728",
            "extra": "mean: 4.710562411200004 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "60844986+larsevj@users.noreply.github.com",
            "name": "Lars Evje",
            "username": "larsevj"
          },
          "distinct": true,
          "id": "1a4c078c4ec040d207c453c20f69f97f08ef9571",
          "message": "Bump codecov/codecov-action from 4 to 5\n\nBumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 4 to 5.\n- [Release notes](https://github.com/codecov/codecov-action/releases)\n- [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/codecov/codecov-action/compare/v4...v5)\n\n---\nupdated-dependencies:\n- dependency-name: codecov/codecov-action\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>\n\nClean up files left by codecov workflow",
          "timestamp": "2025-01-06T14:16:52+01:00",
          "tree_id": "c03557837eb4464ad156112281889ec6eb4be452",
          "url": "https://github.com/equinor/ert/commit/1a4c078c4ec040d207c453c20f69f97f08ef9571"
        },
        "date": 1736169540972,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21421863391165544,
            "unit": "iter/sec",
            "range": "stddev: 0.03453731157924506",
            "extra": "mean: 4.6681279856 sec\nrounds: 5"
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
          "id": "6c9db0b556ce39be3e676bc2b4337895d3477f8e",
          "message": "Do not run actionlint twice on PRs",
          "timestamp": "2025-01-06T14:23:45+01:00",
          "tree_id": "d3869c39a14d10db0b96185a692745e692e36717",
          "url": "https://github.com/equinor/ert/commit/6c9db0b556ce39be3e676bc2b4337895d3477f8e"
        },
        "date": 1736169933337,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21983364004082406,
            "unit": "iter/sec",
            "range": "stddev: 0.03101188336562572",
            "extra": "mean: 4.548894335799997 sec\nrounds: 5"
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
          "id": "d24f588722a7a1e101a6aa8827e9fec6b933a873",
          "message": "Add post/pre experiment simulation hooks\n\n* Add post/pre experiment simulation hooks\r\n* Add docs for PRE/POST_EXPERIMENT hooks",
          "timestamp": "2025-01-06T14:39:24+01:00",
          "tree_id": "7e5efe7fe0c93b33cbe66f3d0d5f78da13460574",
          "url": "https://github.com/equinor/ert/commit/d24f588722a7a1e101a6aa8827e9fec6b933a873"
        },
        "date": 1736170880245,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21887509448985565,
            "unit": "iter/sec",
            "range": "stddev: 0.02236606260307451",
            "extra": "mean: 4.5688158460000015 sec\nrounds: 5"
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
          "id": "790933d7b29bc68b13c2534fea91bc9ad58911c8",
          "message": "Delete duplicate enum",
          "timestamp": "2025-01-06T15:11:29+01:00",
          "tree_id": "f92c72ac4373b2fe3a8a81c7fbeadffc9e15fdcc",
          "url": "https://github.com/equinor/ert/commit/790933d7b29bc68b13c2534fea91bc9ad58911c8"
        },
        "date": 1736172798660,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2182612177777976,
            "unit": "iter/sec",
            "range": "stddev: 0.024209504476434666",
            "extra": "mean: 4.581665997200003 sec\nrounds: 5"
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
          "id": "3f8b3206db9ec63b09af372802998768f61c7adc",
          "message": "Upgrade pre-commits ruff 0.8.3 -> 0.8.6",
          "timestamp": "2025-01-06T16:49:48+01:00",
          "tree_id": "f008e746dadb4dc2d7041b8da37b90a12e993f01",
          "url": "https://github.com/equinor/ert/commit/3f8b3206db9ec63b09af372802998768f61c7adc"
        },
        "date": 1736178696963,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21906439593749036,
            "unit": "iter/sec",
            "range": "stddev: 0.049432446890215",
            "extra": "mean: 4.564867767399994 sec\nrounds: 5"
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
          "id": "a6d2dba8cd9c5cd9368f1b891931b7982c438051",
          "message": "Replace custom schema generation with library",
          "timestamp": "2025-01-07T09:17:04+01:00",
          "tree_id": "b59bfc8de678bde5e848ae7a43293857e884ee9d",
          "url": "https://github.com/equinor/ert/commit/a6d2dba8cd9c5cd9368f1b891931b7982c438051"
        },
        "date": 1736237936574,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21600439227720425,
            "unit": "iter/sec",
            "range": "stddev: 0.013815307542831789",
            "extra": "mean: 4.629535489800008 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}