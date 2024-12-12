window.BENCHMARK_DATA = {
  "lastUpdate": 1734002782754,
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
          "id": "b903bb03be7e401622f604442691b36e5addbde4",
          "message": "Remove overriding of tempfile.tempdir\n\nWhen the jobs are executed on the cluster, /user/run/<userid> is not set up,\nthough XDG_RUNTIME_DIR points to it. This is not a problem for ert as it\nruns the main application locally, but is a problem for Everest where the\nmain application runs on the cluster. So the way lsf logs in to the node is the reason.",
          "timestamp": "2024-12-05T21:13:16+01:00",
          "tree_id": "656ff4f7053e3f065b481da16cf0864a80775c79",
          "url": "https://github.com/equinor/ert/commit/b903bb03be7e401622f604442691b36e5addbde4"
        },
        "date": 1733429707584,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21689946115874717,
            "unit": "iter/sec",
            "range": "stddev: 0.04384166878033591",
            "extra": "mean: 4.610431001799986 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "03903bbc50fbd911313469a10d29c4d43636917c",
          "message": "Improve exit code detection in the slurm driver (#9440)\n\nImprove exit code detection in the slurm driver",
          "timestamp": "2024-12-06T10:27:57+01:00",
          "tree_id": "e21e8c5f40dffa7bee9d0754cc27584a9846b319",
          "url": "https://github.com/equinor/ert/commit/03903bbc50fbd911313469a10d29c4d43636917c"
        },
        "date": 1733477386342,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21656208453872952,
            "unit": "iter/sec",
            "range": "stddev: 0.028287705830547526",
            "extra": "mean: 4.6176134761999945 sec\nrounds: 5"
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
          "id": "aa45608cf666c28b38c32d6b339fe9ee04cf6cb9",
          "message": "Move addition of activate script",
          "timestamp": "2024-12-06T10:38:26+01:00",
          "tree_id": "0e5558d1ba63b8c98d1191b1779ffa407a4459f3",
          "url": "https://github.com/equinor/ert/commit/aa45608cf666c28b38c32d6b339fe9ee04cf6cb9"
        },
        "date": 1733478014035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21382553020849498,
            "unit": "iter/sec",
            "range": "stddev: 0.05113461513636054",
            "extra": "mean: 4.67671002159999 sec\nrounds: 5"
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
          "id": "5061a9426cc6f5ae2db17d2f958a6c9ecfe0cb58",
          "message": "Inline redefinition of components in test",
          "timestamp": "2024-12-06T12:34:23+01:00",
          "tree_id": "e44e579d3509cb6d869a7ded31288311d70c227f",
          "url": "https://github.com/equinor/ert/commit/5061a9426cc6f5ae2db17d2f958a6c9ecfe0cb58"
        },
        "date": 1733484970869,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21343338675248258,
            "unit": "iter/sec",
            "range": "stddev: 0.059976100777467865",
            "extra": "mean: 4.685302591200008 sec\nrounds: 5"
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
          "id": "710e6d5d13a16ec1df4340910b52ca6328f3b6bb",
          "message": "Migrate finalized keys for response configs",
          "timestamp": "2024-12-06T12:34:42+01:00",
          "tree_id": "d5c0168545e3cc674dbc6545b05a0c8189767d37",
          "url": "https://github.com/equinor/ert/commit/710e6d5d13a16ec1df4340910b52ca6328f3b6bb"
        },
        "date": 1733484987413,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21560824691593727,
            "unit": "iter/sec",
            "range": "stddev: 0.029463112519569366",
            "extra": "mean: 4.638041514199995 sec\nrounds: 5"
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
          "id": "4b34fc94ceab52c8ad34e5059837b7ec26e5dcb9",
          "message": "Delete unused `evaluate` method in ensemble_evaluator_utils",
          "timestamp": "2024-12-06T12:34:46+01:00",
          "tree_id": "3f250a19d7c5ac087dd71f3e56a1c1141feeaa7a",
          "url": "https://github.com/equinor/ert/commit/4b34fc94ceab52c8ad34e5059837b7ec26e5dcb9"
        },
        "date": 1733484995231,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.208689645651837,
            "unit": "iter/sec",
            "range": "stddev: 0.05284187912160273",
            "extra": "mean: 4.791804580799993 sec\nrounds: 5"
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
          "id": "726a273e44984d59214426751203284b10d27efe",
          "message": "Have ModelConfig output more noticable warning when malformatted runpath\n\nThis commit makes ModelConfig emit a ConfigWarning if the input runpath does not contain `<ITER>` or `<IENS>`. This was previously only a warning in the logs, but it should be more noticable.",
          "timestamp": "2024-12-06T12:50:39+01:00",
          "tree_id": "9a6f2404d6a597848fafec7c30f82ba0d17c5fe1",
          "url": "https://github.com/equinor/ert/commit/726a273e44984d59214426751203284b10d27efe"
        },
        "date": 1733485947918,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21179165591509186,
            "unit": "iter/sec",
            "range": "stddev: 0.06572133260757662",
            "extra": "mean: 4.7216213295999925 sec\nrounds: 5"
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
          "id": "90c14a927fb0e4c1de9e08a88f7599f329342a0a",
          "message": "Improve UX for permission errors in storage\n\nThis commit:\n* Improves the error message displayed when the dark storage server does not have access to the storage path.\n* Makes the dark storage server return a response with status code 401 - unauthorized when the `get_ensemble_record` endpoint fails due to `PermissionError`.\n* Makes the failed message in `LegacyEnsemble._evaluate_inner` omit stacktrace when it failed due to PermissionError, making it shorter and more consise.",
          "timestamp": "2024-12-06T12:51:10+01:00",
          "tree_id": "5c8c3d177362548299096fd3e12b750c461d7570",
          "url": "https://github.com/equinor/ert/commit/90c14a927fb0e4c1de9e08a88f7599f329342a0a"
        },
        "date": 1733485981654,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2137260034404691,
            "unit": "iter/sec",
            "range": "stddev: 0.05363075399486037",
            "extra": "mean: 4.678887846599997 sec\nrounds: 5"
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
            "email": "frodeaarstad@gmail.com",
            "name": "Frode Aarstad",
            "username": "frode-aarstad"
          },
          "distinct": true,
          "id": "baeb4f52bbbf45a292b0ca943d700c4d485c6c5b",
          "message": "Update everest snapshot egg-py311.csv",
          "timestamp": "2024-12-06T18:42:55+01:00",
          "tree_id": "586c7b283858f837abcdc0c627535b72d1accfed",
          "url": "https://github.com/equinor/ert/commit/baeb4f52bbbf45a292b0ca943d700c4d485c6c5b"
        },
        "date": 1733507089070,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21517625560161308,
            "unit": "iter/sec",
            "range": "stddev: 0.07914358763160681",
            "extra": "mean: 4.647352921000004 sec\nrounds: 5"
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
          "id": "a5bfbdb3e14fd6eaf0155dc545fdcb8c1ba9b2d3",
          "message": "Mute marginal cpu overspending\n\nThere exists logs that a user has overspent with a factor of 1.0. This is not very\ninteresting, so skip logging anything that we don't find significant.",
          "timestamp": "2024-12-09T12:54:57+01:00",
          "tree_id": "531f7b1e539f789dc17711ef32c2ede1cad32ff9",
          "url": "https://github.com/equinor/ert/commit/a5bfbdb3e14fd6eaf0155dc545fdcb8c1ba9b2d3"
        },
        "date": 1733745408650,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22054059747100946,
            "unit": "iter/sec",
            "range": "stddev: 0.021143488419735627",
            "extra": "mean: 4.534312555000002 sec\nrounds: 5"
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
          "id": "ea78bd8890343d1c99337be738c3ccae8c32fd45",
          "message": "Only set active realizations from selected ensemble when restart is checked es_mda",
          "timestamp": "2024-12-09T14:06:59+01:00",
          "tree_id": "07962ecd11797d4dc9a46e969fda438d1acbeead",
          "url": "https://github.com/equinor/ert/commit/ea78bd8890343d1c99337be738c3ccae8c32fd45"
        },
        "date": 1733749733640,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21473519569661298,
            "unit": "iter/sec",
            "range": "stddev: 0.09679680966034868",
            "extra": "mean: 4.656898449999983 sec\nrounds: 5"
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
          "id": "0bbd49e3a867f28f6de840d210336ba881d17405",
          "message": "Require positive std_cutoff",
          "timestamp": "2024-12-09T14:50:11+01:00",
          "tree_id": "60ed741b163f4a1fae7958406e222f85a4804ec6",
          "url": "https://github.com/equinor/ert/commit/0bbd49e3a867f28f6de840d210336ba881d17405"
        },
        "date": 1733752319912,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2140253415088439,
            "unit": "iter/sec",
            "range": "stddev: 0.02019763556109462",
            "extra": "mean: 4.672343905399998 sec\nrounds: 5"
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
          "id": "ad786d8aafd2a68307598f251b74f4f415615bc2",
          "message": "Use pattern matching for make_summary_key",
          "timestamp": "2024-12-09T15:35:09+01:00",
          "tree_id": "792a5098ecdd3cd0f48f7780d324b0eaa547cae3",
          "url": "https://github.com/equinor/ert/commit/ad786d8aafd2a68307598f251b74f4f415615bc2"
        },
        "date": 1733755018491,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21490196426339978,
            "unit": "iter/sec",
            "range": "stddev: 0.018610687178640374",
            "extra": "mean: 4.653284596199995 sec\nrounds: 5"
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
          "id": "35e410144fdcee7c64e5675be461cdeacccef98f",
          "message": "Use pattern-matching for ensemble_parameters",
          "timestamp": "2024-12-09T15:34:44+01:00",
          "tree_id": "fd510197e7304a33062b38a6d9f7b9311e7ec812",
          "url": "https://github.com/equinor/ert/commit/35e410144fdcee7c64e5675be461cdeacccef98f"
        },
        "date": 1733755046896,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21639733693184626,
            "unit": "iter/sec",
            "range": "stddev: 0.03895051510979895",
            "extra": "mean: 4.621128957400003 sec\nrounds: 5"
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
          "id": "8e003395f1047236edfa2675279cdf16e42ef593",
          "message": "Stop using \"cached\" testdata in test_export.py",
          "timestamp": "2024-12-10T08:42:32+01:00",
          "tree_id": "f9de49f0e2c763abbc8071520bfd18ab6941a190",
          "url": "https://github.com/equinor/ert/commit/8e003395f1047236edfa2675279cdf16e42ef593"
        },
        "date": 1733816662019,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22015221654480238,
            "unit": "iter/sec",
            "range": "stddev: 0.044098889337104534",
            "extra": "mean: 4.542311749999999 sec\nrounds: 5"
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
          "id": "bf2d515c9e9948fcad5caa4b4fba7c60e6339cfa",
          "message": "Fix ForwardModelStep `handle_process_timeout...` timeout\n\nThis fixes the bug where some code was unreachable after the refactoring in commit #4dc894ca63687476e091f582df5a42045190f7bd",
          "timestamp": "2024-12-10T09:18:33+01:00",
          "tree_id": "fac3444a941125e008dfefb9b8a03fbedecc6620",
          "url": "https://github.com/equinor/ert/commit/bf2d515c9e9948fcad5caa4b4fba7c60e6339cfa"
        },
        "date": 1733818824915,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2214792527246713,
            "unit": "iter/sec",
            "range": "stddev: 0.045115848454224346",
            "extra": "mean: 4.515095602399993 sec\nrounds: 5"
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
          "id": "602a9b2444fe6262514f7a76665d4d4237e1b498",
          "message": "Use pattern matching for config schema",
          "timestamp": "2024-12-10T09:49:48+01:00",
          "tree_id": "ab22c6111029f34e88c76e7578a83f1e6169d31c",
          "url": "https://github.com/equinor/ert/commit/602a9b2444fe6262514f7a76665d4d4237e1b498"
        },
        "date": 1733820704060,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2183465894258909,
            "unit": "iter/sec",
            "range": "stddev: 0.026407073519232785",
            "extra": "mean: 4.579874605000003 sec\nrounds: 5"
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
          "id": "a156a6b176ae1d5ebdf1b6f3b7b1131671f67b7f",
          "message": "Mark tests taking more than a second as integration test",
          "timestamp": "2024-12-10T10:47:43+01:00",
          "tree_id": "4f9bac00943bca26cb1c724a1233d7ea809d16ae",
          "url": "https://github.com/equinor/ert/commit/a156a6b176ae1d5ebdf1b6f3b7b1131671f67b7f"
        },
        "date": 1733824172016,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22711674495822548,
            "unit": "iter/sec",
            "range": "stddev: 0.03677181838482849",
            "extra": "mean: 4.403021891599996 sec\nrounds: 5"
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
          "id": "4efb1902a3f15cbdd877c89d5f2865f857f066e6",
          "message": "Remove code paths for python <3.11",
          "timestamp": "2024-12-10T10:50:20+01:00",
          "tree_id": "cdb466a57f6a494ae86e0787ce809dc98f75045c",
          "url": "https://github.com/equinor/ert/commit/4efb1902a3f15cbdd877c89d5f2865f857f066e6"
        },
        "date": 1733824327122,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21451582590554963,
            "unit": "iter/sec",
            "range": "stddev: 0.08362406842782452",
            "extra": "mean: 4.661660722599999 sec\nrounds: 5"
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
          "id": "60fa1a6e1bd598aa5569820698a9801f3e79f329",
          "message": "Fix bug in careful_copy handling None\n\nThe function careful_copy_file was not able to handle None as\na target. Added the functionality as probably originally intended\nand added tests.",
          "timestamp": "2024-12-10T11:31:06+01:00",
          "tree_id": "c856d2dec03203e486342d9fc3e5e1ebcb04f2e3",
          "url": "https://github.com/equinor/ert/commit/60fa1a6e1bd598aa5569820698a9801f3e79f329"
        },
        "date": 1733826772370,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22661602393836813,
            "unit": "iter/sec",
            "range": "stddev: 0.026439295745565545",
            "extra": "mean: 4.412750619400003 sec\nrounds: 5"
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
          "id": "68d01c72cb386db9c9236591c52b3862d12ad375",
          "message": "Lower test_memory_smoothing threshold",
          "timestamp": "2024-12-10T11:53:29+01:00",
          "tree_id": "c9197c37daaa4da06cb25d941da7998420010f10",
          "url": "https://github.com/equinor/ert/commit/68d01c72cb386db9c9236591c52b3862d12ad375"
        },
        "date": 1733828121140,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21975888542340732,
            "unit": "iter/sec",
            "range": "stddev: 0.029186475432132337",
            "extra": "mean: 4.550441717400003 sec\nrounds: 5"
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
          "id": "e48000189c43824c95b4daea5de139435394b056",
          "message": "Improve UX for permission errors in storage\n\nThis commit:\n* Improves the error message displayed when the dark storage server does not have access to the storage path.\n* Makes the dark storage server return a response with status code 401 - unauthorized when the `get_ensemble_record` endpoint fails due to `PermissionError`.\n* Makes the failed message in `LegacyEnsemble._evaluate_inner` omit stacktrace when it failed due to PermissionError, making it shorter and more consise.",
          "timestamp": "2024-12-10T13:27:35+01:00",
          "tree_id": "05cf1f88473346617a2e927b73a5f0d8c3f0b0c9",
          "url": "https://github.com/equinor/ert/commit/e48000189c43824c95b4daea5de139435394b056"
        },
        "date": 1733833767805,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21821189918885073,
            "unit": "iter/sec",
            "range": "stddev: 0.034685910678909984",
            "extra": "mean: 4.582701510400005 sec\nrounds: 5"
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
          "id": "ae944073bcd130adf03cd3d12235cee722941622",
          "message": "Remove unused mock validate_active_realizations_count-method in tests for runmodel",
          "timestamp": "2024-12-10T13:27:54+01:00",
          "tree_id": "4b28eb07d72c3742c513a10a2f7a10cb40feb1d1",
          "url": "https://github.com/equinor/ert/commit/ae944073bcd130adf03cd3d12235cee722941622"
        },
        "date": 1733833789077,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.209538826607953,
            "unit": "iter/sec",
            "range": "stddev: 0.035846505022795204",
            "extra": "mean: 4.772385224199996 sec\nrounds: 5"
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
          "id": "82972b0db0244aa4787b1a8385fb4280d8f4fc6b",
          "message": "Remove use of deprecated types",
          "timestamp": "2024-12-10T13:54:24+01:00",
          "tree_id": "1931f5ebec992e066badc2a18049bc4b912160db",
          "url": "https://github.com/equinor/ert/commit/82972b0db0244aa4787b1a8385fb4280d8f4fc6b"
        },
        "date": 1733835370227,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22017516807077772,
            "unit": "iter/sec",
            "range": "stddev: 0.013477091318217439",
            "extra": "mean: 4.541838249799992 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "60844986+larsevj@users.noreply.github.com",
            "name": "Lars Evje",
            "username": "larsevj"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9d43bc005ac7a77564372e3ef80a9c6ce7ab567",
          "message": "Add pyupgrade UP rule to ruff\n\n---------\r\n\r\nCo-authored-by: Eivind Jahren <ejah@equinor.com>",
          "timestamp": "2024-12-10T20:42:32Z",
          "tree_id": "4ce4a43347adab0c59907dd8c59683c70757f538",
          "url": "https://github.com/equinor/ert/commit/c9d43bc005ac7a77564372e3ef80a9c6ce7ab567"
        },
        "date": 1733863468744,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21071038331313507,
            "unit": "iter/sec",
            "range": "stddev: 0.07808106962564963",
            "extra": "mean: 4.74585060440001 sec\nrounds: 5"
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
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "dffa41ef351c87292a0291e4d5c1c0036b41c8e8",
          "message": "Run pyupgrade on src tree",
          "timestamp": "2024-12-11T09:28:00+01:00",
          "tree_id": "45c02de10652357c09899afad6c9adaeea84adb4",
          "url": "https://github.com/equinor/ert/commit/dffa41ef351c87292a0291e4d5c1c0036b41c8e8"
        },
        "date": 1733905814889,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21846783445621357,
            "unit": "iter/sec",
            "range": "stddev: 0.012075998138562862",
            "extra": "mean: 4.5773328714 sec\nrounds: 5"
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
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "c9ae9b2844386c9930d70ae003e6fc30a65cbefe",
          "message": "Increase esupdate performance test limits",
          "timestamp": "2024-12-11T09:44:49+01:00",
          "tree_id": "6504f3b093cd044251368b59edaeb27bc7396ab1",
          "url": "https://github.com/equinor/ert/commit/c9ae9b2844386c9930d70ae003e6fc30a65cbefe"
        },
        "date": 1733906821039,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22206886758150107,
            "unit": "iter/sec",
            "range": "stddev: 0.014318287483247165",
            "extra": "mean: 4.503107576000008 sec\nrounds: 5"
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
          "id": "05b33a5039bc8135ce171323b2eb6dc180230534",
          "message": "Remove deprecated comment,\n\nref https://github.com/equinor/ert/pull/9495#discussion_r1880099231",
          "timestamp": "2024-12-11T13:56:28+01:00",
          "tree_id": "5eebbc49e599afe5187c4c6bcdd258111843cd15",
          "url": "https://github.com/equinor/ert/commit/05b33a5039bc8135ce171323b2eb6dc180230534"
        },
        "date": 1733921919420,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2172208094816743,
            "unit": "iter/sec",
            "range": "stddev: 0.03218128145278603",
            "extra": "mean: 4.60361050300001 sec\nrounds: 5"
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
          "id": "a04e2df77855edab07e947cc02d34dcde14dd173",
          "message": "Replace usage of deprecated logging.warn with logging.warning",
          "timestamp": "2024-12-12T17:19:37+09:00",
          "tree_id": "5c5df4860ad457647d5b22debf651c5fdcdedfd7",
          "url": "https://github.com/equinor/ert/commit/a04e2df77855edab07e947cc02d34dcde14dd173"
        },
        "date": 1733991695350,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22181901268364088,
            "unit": "iter/sec",
            "range": "stddev: 0.04076522473732182",
            "extra": "mean: 4.508179835000004 sec\nrounds: 5"
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
          "id": "104c326981cc4ccfe163c805ceac4b01398c9262",
          "message": "Small logic fix in data_for_key",
          "timestamp": "2024-12-12T12:24:35+01:00",
          "tree_id": "d6dd86579308f60f5e7ba8446206947827d73ed7",
          "url": "https://github.com/equinor/ert/commit/104c326981cc4ccfe163c805ceac4b01398c9262"
        },
        "date": 1734002782290,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22281022911173926,
            "unit": "iter/sec",
            "range": "stddev: 0.03272720269613646",
            "extra": "mean: 4.488124283999997 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}