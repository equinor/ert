window.BENCHMARK_DATA = {
  "lastUpdate": 1736930832962,
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
          "id": "1ae12f6efb66e9fa4208faef2a04898043cdcb37",
          "message": "Replace erts interface towards reservoir simulators\n\nThis replaces the yaml configuration file for Eclipse100/300 with a set\nof environment variables set through the plugin system. Ert cannot any\nlonger start the raw Eclipse binary itself, it depends on the vendor\nsupplied wrapper binary called \"eclrun\".\n\nSimilarly, for OPM flow, Ert will now support a wrapper script \"flowrun\"\nif it is present, assuming it has a similar command line API as eclrun.\nIf flowrun is not present, it will look for a binary \"flow\" in $PATH\nwhich can be used, but then only with single-cpu possibilities.\n\nUsers can point to a custom location of eclrun by adding SETENV to the\nconfiguration file.",
          "timestamp": "2025-01-09T16:40:43+01:00",
          "tree_id": "3e58eb10aacae19e6e717c57d30f333d00533dac",
          "url": "https://github.com/equinor/ert/commit/1ae12f6efb66e9fa4208faef2a04898043cdcb37"
        },
        "date": 1736437375543,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2168821050902408,
            "unit": "iter/sec",
            "range": "stddev: 0.04013748871607781",
            "extra": "mean: 4.610799953200001 sec\nrounds: 5"
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
          "id": "7b2bedf144cc86b1fcaef3e5b880522c8cf9a384",
          "message": "Show inversion errors in GUI",
          "timestamp": "2025-01-10T08:46:43+01:00",
          "tree_id": "3131ea715763f6eeed1c20e72e0b88787dfcc076",
          "url": "https://github.com/equinor/ert/commit/7b2bedf144cc86b1fcaef3e5b880522c8cf9a384"
        },
        "date": 1736495316715,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21176326382634605,
            "unit": "iter/sec",
            "range": "stddev: 0.029305271839122727",
            "extra": "mean: 4.722254379399999 sec\nrounds: 5"
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
          "id": "2a3f4f041b6b7d4c1bd022d9259003518ee9a534",
          "message": "Ensure pre_experiment validation has access to plugin configuration\n\nThis is a fixup of a regression in 1ae12f6efb66e9fa4208faef2a04898043cdcb37\n\nThe bug slipped through as the feature is skipped unless the test is\nrun on-premise",
          "timestamp": "2025-01-10T13:24:24+01:00",
          "tree_id": "1892de1b99f8307950852f4834c1616e24ed2bef",
          "url": "https://github.com/equinor/ert/commit/2a3f4f041b6b7d4c1bd022d9259003518ee9a534"
        },
        "date": 1736511980458,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.219106216927792,
            "unit": "iter/sec",
            "range": "stddev: 0.03126683911471407",
            "extra": "mean: 4.563996467200002 sec\nrounds: 5"
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
          "id": "d8338514a2b3c9cbe6c13073631ac6a605115af0",
          "message": "Reorganize workflow tests",
          "timestamp": "2025-01-10T13:53:00+01:00",
          "tree_id": "f6bbb6a890b146da22f7cb61f401620f0973fbfd",
          "url": "https://github.com/equinor/ert/commit/d8338514a2b3c9cbe6c13073631ac6a605115af0"
        },
        "date": 1736513706788,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21239781532714902,
            "unit": "iter/sec",
            "range": "stddev: 0.13264117300765876",
            "extra": "mean: 4.708146354799998 sec\nrounds: 5"
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
          "id": "4480295150a8db352a5bf1020388ec5f0c60a002",
          "message": "Handle button visual indications based on actions\n\nUse SidebarToolButton instead of QToolButton",
          "timestamp": "2025-01-10T14:31:01+01:00",
          "tree_id": "819cb21a0a52f945c26565a7c0c3ec336f3085f6",
          "url": "https://github.com/equinor/ert/commit/4480295150a8db352a5bf1020388ec5f0c60a002"
        },
        "date": 1736515969471,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21855746435616563,
            "unit": "iter/sec",
            "range": "stddev: 0.03610331598553161",
            "extra": "mean: 4.575455718000003 sec\nrounds: 5"
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
          "id": "ea36931b29b232b311324947ef9da1dd934fa75f",
          "message": "Refactor preferred_num_cpu",
          "timestamp": "2025-01-13T09:16:12+01:00",
          "tree_id": "828f74b8c8cb2f1add0bdf6aec383e0a13fe0627",
          "url": "https://github.com/equinor/ert/commit/ea36931b29b232b311324947ef9da1dd934fa75f"
        },
        "date": 1736756287297,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21666479591781931,
            "unit": "iter/sec",
            "range": "stddev: 0.047548023697719474",
            "extra": "mean: 4.615424465999999 sec\nrounds: 5"
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
          "id": "5a0ff5f28e60d171200bf52d919161ffc806fe1a",
          "message": "Adapt everest to run_reservoirsimulator\n\nThis is a fixup of a regression from 1ae12f6efb66e9fa4208faef2a04898043cdcb37",
          "timestamp": "2025-01-13T09:32:40+01:00",
          "tree_id": "5dbdb935040aecc63ba8359a2291701ea8123fdb",
          "url": "https://github.com/equinor/ert/commit/5a0ff5f28e60d171200bf52d919161ffc806fe1a"
        },
        "date": 1736757268225,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2182373089808161,
            "unit": "iter/sec",
            "range": "stddev: 0.031243100457874057",
            "extra": "mean: 4.5821679375999995 sec\nrounds: 5"
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
          "id": "051fa313ae1ce02993516d4a8151b840312bebea",
          "message": "Refactor update workflow signature",
          "timestamp": "2025-01-13T11:02:50+01:00",
          "tree_id": "cbb28345d636245f0bf73ef7166f305550924bf3",
          "url": "https://github.com/equinor/ert/commit/051fa313ae1ce02993516d4a8151b840312bebea"
        },
        "date": 1736762682429,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2191419788233464,
            "unit": "iter/sec",
            "range": "stddev: 0.028790478476342574",
            "extra": "mean: 4.563251666199998 sec\nrounds: 5"
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
          "id": "bd9cb3ade4e128625749f9a087d792ed5b1a0c4f",
          "message": "Upgrade pre-commits ruff 0.8.6 -> 0.9.1",
          "timestamp": "2025-01-13T13:04:35+01:00",
          "tree_id": "1ed1cb7c04e0e3f6ffe046c908063bfd54256a29",
          "url": "https://github.com/equinor/ert/commit/bd9cb3ade4e128625749f9a087d792ed5b1a0c4f"
        },
        "date": 1736769990385,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21615649579850868,
            "unit": "iter/sec",
            "range": "stddev: 0.024690762932409144",
            "extra": "mean: 4.626277810000005 sec\nrounds: 5"
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
          "id": "718fd5046d5b52271b5bc58a5d1c366b4bf8ecb5",
          "message": "Rename test_jobs_file_is_backed_up",
          "timestamp": "2025-01-13T13:23:57+01:00",
          "tree_id": "64a98a737d415a2b1631ae1037f5ad22086e768b",
          "url": "https://github.com/equinor/ert/commit/718fd5046d5b52271b5bc58a5d1c366b4bf8ecb5"
        },
        "date": 1736771153276,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2164381185900452,
            "unit": "iter/sec",
            "range": "stddev: 0.02403586279174922",
            "extra": "mean: 4.620258235999995 sec\nrounds: 5"
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
          "id": "ef7daed36c880922f1b7ff23904943e6bd72a18d",
          "message": "Fix missing comma in SUMMARY_KEYS everest",
          "timestamp": "2025-01-13T13:33:48+01:00",
          "tree_id": "1495932fa477fe6dbc8c9b1b8f24383b8375088f",
          "url": "https://github.com/equinor/ert/commit/ef7daed36c880922f1b7ff23904943e6bd72a18d"
        },
        "date": 1736771737563,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21784676105122167,
            "unit": "iter/sec",
            "range": "stddev: 0.03495697133700672",
            "extra": "mean: 4.590382685400004 sec\nrounds: 5"
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
          "id": "5fb2308f9e5acbfcdb59d4c314eebfc2aa5cd543",
          "message": "Only test for mac when pushing changes to main branch\n\nRemove `macos-13` since this is excluded for specified python version\nChange `macos-14` to `macos-latest` as we want to build for arm-architecture,\nand also this aligns with our free CI-minutes\nRemove `macos-14-large` as testing also on `x64` architecture seems excessive.",
          "timestamp": "2025-01-13T13:40:36+01:00",
          "tree_id": "49c5192f2b2088dc8946972fd600b4572a446681",
          "url": "https://github.com/equinor/ert/commit/5fb2308f9e5acbfcdb59d4c314eebfc2aa5cd543"
        },
        "date": 1736772148875,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21719920321136096,
            "unit": "iter/sec",
            "range": "stddev: 0.02350508586191256",
            "extra": "mean: 4.604068455199993 sec\nrounds: 5"
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
          "id": "5cb786131c2fa649f06590a6a3981827ede2aa7c",
          "message": "Ignore OSErrors on subprocess call in poll() and bhist\n\nPretend these kinds of issues are flaky. It is important not to crash on\npotentially intermittent failures in code that is rerun every 2 seconds.",
          "timestamp": "2025-01-13T14:12:57+01:00",
          "tree_id": "0e0ee7ca7b9f8b86bc68ea36a81f6122a2756eee",
          "url": "https://github.com/equinor/ert/commit/5cb786131c2fa649f06590a6a3981827ede2aa7c"
        },
        "date": 1736774089965,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2104152307761235,
            "unit": "iter/sec",
            "range": "stddev: 0.030580872338815884",
            "extra": "mean: 4.752507678799995 sec\nrounds: 5"
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
          "id": "7368dc8b6c2cebaf16cfd1719b7a67fbb52b63f7",
          "message": "Remove unused install_dependencies action",
          "timestamp": "2025-01-13T14:16:58+01:00",
          "tree_id": "e6108455414823c5fbb71bcd191abeb967057ce5",
          "url": "https://github.com/equinor/ert/commit/7368dc8b6c2cebaf16cfd1719b7a67fbb52b63f7"
        },
        "date": 1736774297851,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21706520962393724,
            "unit": "iter/sec",
            "range": "stddev: 0.04341485545408905",
            "extra": "mean: 4.6069105304 sec\nrounds: 5"
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
          "id": "9b0a3b198d25475335e3b4cdd1543511ca2db3f4",
          "message": "Use pattern matching",
          "timestamp": "2025-01-13T14:28:24+01:00",
          "tree_id": "bcebbde614f747722f06616461b473476bf23e6c",
          "url": "https://github.com/equinor/ert/commit/9b0a3b198d25475335e3b4cdd1543511ca2db3f4"
        },
        "date": 1736774986785,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2179556951335767,
            "unit": "iter/sec",
            "range": "stddev: 0.014412422697833875",
            "extra": "mean: 4.5880884158000015 sec\nrounds: 5"
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
          "id": "94439f500d3885e8fbe25809650defffb7c8ea57",
          "message": "Make scheduler cancel() and kill_all_jobs async\n\nThis commit makes the methods async, so we can await them instead\nfire-and-forgetting them through asyncio.run_coroutine_threadsafe.",
          "timestamp": "2025-01-13T15:09:36+01:00",
          "tree_id": "8396fa683811a3db413af1ef300899f9896063f5",
          "url": "https://github.com/equinor/ert/commit/94439f500d3885e8fbe25809650defffb7c8ea57"
        },
        "date": 1736777456634,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22045692267310796,
            "unit": "iter/sec",
            "range": "stddev: 0.02251458215253986",
            "extra": "mean: 4.536033561000002 sec\nrounds: 5"
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
          "id": "be7f637eedc5eb1f434d3f5564de315874712775",
          "message": "Bump astral-sh/setup-uv from 4 to 5\n\nBumps [astral-sh/setup-uv](https://github.com/astral-sh/setup-uv) from 4 to 5.\n- [Release notes](https://github.com/astral-sh/setup-uv/releases)\n- [Commits](https://github.com/astral-sh/setup-uv/compare/v4...v5)\n\n---\nupdated-dependencies:\n- dependency-name: astral-sh/setup-uv\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2025-01-13T18:53:29+01:00",
          "tree_id": "ca09298659ce5c2be17a248534476e7b476d2e6b",
          "url": "https://github.com/equinor/ert/commit/be7f637eedc5eb1f434d3f5564de315874712775"
        },
        "date": 1736790885338,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.221652605260119,
            "unit": "iter/sec",
            "range": "stddev: 0.03259297945375952",
            "extra": "mean: 4.511564386200002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hsoho@equinor.com",
            "name": "Håkon Steinkopf Søhoel",
            "username": "HakonSohoel"
          },
          "committer": {
            "email": "49289030+HakonSohoel@users.noreply.github.com",
            "name": "Håkon Steinkopf Søhoel",
            "username": "HakonSohoel"
          },
          "distinct": true,
          "id": "735ea0eea127a7e31c7f9fdd2e361316e23d8d61",
          "message": "Set span status if realization fails",
          "timestamp": "2025-01-14T07:41:21+01:00",
          "tree_id": "0c3c3cedd5b96f9ac78a3614b8be7eede0acd97a",
          "url": "https://github.com/equinor/ert/commit/735ea0eea127a7e31c7f9fdd2e361316e23d8d61"
        },
        "date": 1736836953848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22002245427011805,
            "unit": "iter/sec",
            "range": "stddev: 0.028920568117441783",
            "extra": "mean: 4.544990661600002 sec\nrounds: 5"
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
          "id": "9ab687a0f4bcccd9bd23ffa648dae4f3d257828b",
          "message": "Ignore non ConfigWarnings when running everest lint command",
          "timestamp": "2025-01-14T10:18:56+02:00",
          "tree_id": "a6ab75bee14440c3df13e8910e6e5212db1b2c15",
          "url": "https://github.com/equinor/ert/commit/9ab687a0f4bcccd9bd23ffa648dae4f3d257828b"
        },
        "date": 1736842815089,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2162289944848741,
            "unit": "iter/sec",
            "range": "stddev: 0.026282788526628627",
            "extra": "mean: 4.624726681000004 sec\nrounds: 5"
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
          "id": "957b377541ee01bff98910659da0065ae03de820",
          "message": "Fix test when other plugins could be installed",
          "timestamp": "2025-01-14T09:54:33+01:00",
          "tree_id": "822e2ec5f13e98b67f6b0432162f98bd114f0596",
          "url": "https://github.com/equinor/ert/commit/957b377541ee01bff98910659da0065ae03de820"
        },
        "date": 1736844945462,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21686778557003447,
            "unit": "iter/sec",
            "range": "stddev: 0.09525639382765581",
            "extra": "mean: 4.611104398799995 sec\nrounds: 5"
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
          "id": "0c7bf35e9a645989252e4c12c5a81dfbc998b485",
          "message": "Refactor BaseRunModel",
          "timestamp": "2025-01-14T10:19:13+01:00",
          "tree_id": "3d75e935ed5a5b6473d1378774b8f8d6f5a3b2ec",
          "url": "https://github.com/equinor/ert/commit/0c7bf35e9a645989252e4c12c5a81dfbc998b485"
        },
        "date": 1736846428309,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21954249720525112,
            "unit": "iter/sec",
            "range": "stddev: 0.027055588339479842",
            "extra": "mean: 4.554926780600004 sec\nrounds: 5"
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
          "id": "60fc8f83cb00749526374029bf335b577fee9dec",
          "message": "Fix typo in update test",
          "timestamp": "2025-01-14T10:31:07+01:00",
          "tree_id": "87bd243c4d03ad7577f0ffbf79259501779988f5",
          "url": "https://github.com/equinor/ert/commit/60fc8f83cb00749526374029bf335b577fee9dec"
        },
        "date": 1736847139944,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2199965225563273,
            "unit": "iter/sec",
            "range": "stddev: 0.029260392279015925",
            "extra": "mean: 4.5455263946 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "parulek@gmail.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "committer": {
            "email": "jparu@equinor.com",
            "name": "Julius Parulek",
            "username": "xjules"
          },
          "distinct": true,
          "id": "24911859160476b4e16c1e4cae7be912b61a336a",
          "message": "Allow for support for multiple occurences of design matrices\n\n- Add merging strategy for multiple design matrices in the config\n- Add ui tests for multi dm occurence",
          "timestamp": "2025-01-14T13:53:01+01:00",
          "tree_id": "3782ec43fee2972077aa51607532239b38abba0b",
          "url": "https://github.com/equinor/ert/commit/24911859160476b4e16c1e4cae7be912b61a336a"
        },
        "date": 1736859256914,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2139636994132702,
            "unit": "iter/sec",
            "range": "stddev: 0.020972502608077006",
            "extra": "mean: 4.673689989200005 sec\nrounds: 5"
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
          "id": "c43f97fbea72b321e7a500433d96650606493052",
          "message": "Speed up batching of events in evaluator\n\nasyncio wait_for used in evaluator is \"slow\" in performance when dealing with O(100K) events in the event queue.\nTherefore the suggestion is to replace it with direct fetching (via get_nowait())\nand instead just sleep whenever the event queue is empty.\n\nCo-authored-by: Lars Evje <levje@equinor.com>",
          "timestamp": "2025-01-14T14:30:02+01:00",
          "tree_id": "95472962253406a4238a2b9c26f36a40dca0fb49",
          "url": "https://github.com/equinor/ert/commit/c43f97fbea72b321e7a500433d96650606493052"
        },
        "date": 1736861480991,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2156477153113954,
            "unit": "iter/sec",
            "range": "stddev: 0.01899322592247238",
            "extra": "mean: 4.637192648000001 sec\nrounds: 5"
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
          "id": "0d9c178d286cd4647d6a1720196152e6cb4aaa09",
          "message": "Increase Event reporter timeout after sucesful Finish\n\nFor the sake of heavy load when using LOCAL_DRIVER we set the default timeout to 10 minutes since forward model can be finished but not all the events were sent yet.\n\nCo-authored-by: Lars Evje <levje@equinor.com>",
          "timestamp": "2025-01-14T14:30:22+01:00",
          "tree_id": "91739fa37e96e33e0514d8fc01c5be28bb23fa2c",
          "url": "https://github.com/equinor/ert/commit/0d9c178d286cd4647d6a1720196152e6cb4aaa09"
        },
        "date": 1736861496704,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2095919394532994,
            "unit": "iter/sec",
            "range": "stddev: 0.053266134559518064",
            "extra": "mean: 4.771175850600002 sec\nrounds: 5"
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
          "id": "024fcbaa8a960d3a3e8ab6e44ef6e21a38263f9e",
          "message": "Fix bug in docstring of ForwardModelStep",
          "timestamp": "2025-01-14T15:08:21+01:00",
          "tree_id": "d9ffbe775edd3cd4bb44caa76cbd051ace370e92",
          "url": "https://github.com/equinor/ert/commit/024fcbaa8a960d3a3e8ab6e44ef6e21a38263f9e"
        },
        "date": 1736863785139,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21527766177391452,
            "unit": "iter/sec",
            "range": "stddev: 0.03733844344660833",
            "extra": "mean: 4.6451637934000045 sec\nrounds: 5"
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
          "id": "56b33818027d81369ac8a7d2ca0e8d4fafad0ad1",
          "message": "Set the default flow simulator version to default\n\ndaily is too on-premises specific, and will actually not\nwork when the current version of ert-configurations is installed\noff-premise.",
          "timestamp": "2025-01-15T09:22:23+01:00",
          "tree_id": "bb228da62bdff4342e9c0bb6b9293d7eaef996cf",
          "url": "https://github.com/equinor/ert/commit/56b33818027d81369ac8a7d2ca0e8d4fafad0ad1"
        },
        "date": 1736929422661,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2156870193608191,
            "unit": "iter/sec",
            "range": "stddev: 0.030857132655623096",
            "extra": "mean: 4.6363476252 sec\nrounds: 5"
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
          "id": "9a4f5c89d446a262c2b5e631460ca8b64acc9531",
          "message": "Support forwarding multiple arguments to reservoir simulators",
          "timestamp": "2025-01-15T09:23:00+01:00",
          "tree_id": "3bf9434de32b099e82c8e47e369e73cb2ca50197",
          "url": "https://github.com/equinor/ert/commit/9a4f5c89d446a262c2b5e631460ca8b64acc9531"
        },
        "date": 1736929455118,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21827084124134402,
            "unit": "iter/sec",
            "range": "stddev: 0.02584693117079259",
            "extra": "mean: 4.581463993600002 sec\nrounds: 5"
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
          "id": "9a1002d3191d56def21d8c4256184b8af2c9f695",
          "message": "Speed up tests by avoiding recreating PluginManager",
          "timestamp": "2025-01-15T09:31:24+01:00",
          "tree_id": "457b81e89a525dfd80f5833a608c366a39999323",
          "url": "https://github.com/equinor/ert/commit/9a1002d3191d56def21d8c4256184b8af2c9f695"
        },
        "date": 1736929959863,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2145204837317843,
            "unit": "iter/sec",
            "range": "stddev: 0.04616200503930073",
            "extra": "mean: 4.661559505199994 sec\nrounds: 5"
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
          "id": "f60dc0032c2ffe416cfd25bdea17f049828b9d91",
          "message": "Use lalr parser for main ert config\n\nThis is for performance reasons. May cause ambiguous config files\nto no longer be parsed.",
          "timestamp": "2025-01-15T09:45:19+01:00",
          "tree_id": "3b06f6cbbd31bd9c33538727632de61ee9cfb990",
          "url": "https://github.com/equinor/ert/commit/f60dc0032c2ffe416cfd25bdea17f049828b9d91"
        },
        "date": 1736930795080,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.215833648092967,
            "unit": "iter/sec",
            "range": "stddev: 0.03127751551107894",
            "extra": "mean: 4.6331978764000015 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}