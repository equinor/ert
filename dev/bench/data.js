window.BENCHMARK_DATA = {
  "lastUpdate": 1736437376441,
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
      },
      {
        "commit": {
          "author": {
            "email": "havb@equinor.com",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "committer": {
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "23015dc234b1c82de3ac00a5033beea49a4043dc",
          "message": "Avoid UserWarning when vmin==vmax in std_dev plot",
          "timestamp": "2025-01-07T09:24:18+01:00",
          "tree_id": "823ca1fc5e745684f2f8e5bea60439489f9c68fe",
          "url": "https://github.com/equinor/ert/commit/23015dc234b1c82de3ac00a5033beea49a4043dc"
        },
        "date": 1736238367513,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2157662634085902,
            "unit": "iter/sec",
            "range": "stddev: 0.02404537024979024",
            "extra": "mean: 4.63464484299999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "cmrqs@unicamp.br",
            "name": "Carlos Marques",
            "username": "cmrqs"
          },
          "committer": {
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "f1ec937ec63d43140ef8580d26951feb2f6ebaf4",
          "message": "Add unit tests for conda activation script",
          "timestamp": "2025-01-07T09:25:08+01:00",
          "tree_id": "49d1f63f823339ceeac2c36f937e4c9467f82017",
          "url": "https://github.com/equinor/ert/commit/f1ec937ec63d43140ef8580d26951feb2f6ebaf4"
        },
        "date": 1736238422936,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21964482161743082,
            "unit": "iter/sec",
            "range": "stddev: 0.030207119201486343",
            "extra": "mean: 4.552804808400003 sec\nrounds: 5"
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
          "id": "d82fc01bc98f4ea3182b5e7501b255d05bca95b3",
          "message": "Add GEN_KW to heat equation and turn localization on\n\nMakes it easier to write more realistic tests.",
          "timestamp": "2025-01-07T09:26:23+01:00",
          "tree_id": "bebba6cc88eed0709ae785b29c50b8df430f7886",
          "url": "https://github.com/equinor/ert/commit/d82fc01bc98f4ea3182b5e7501b255d05bca95b3"
        },
        "date": 1736238489287,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21789469644185985,
            "unit": "iter/sec",
            "range": "stddev: 0.015257471053639851",
            "extra": "mean: 4.589372831599997 sec\nrounds: 5"
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
          "id": "2f81dd1cfd19ee4e89d36817a4de195aa85bbd59",
          "message": "Add deprecation warning for SIMULATION_JOB keyword",
          "timestamp": "2025-01-07T14:16:32+02:00",
          "tree_id": "f9abd862ea95a48632f06e06889734b53942eb43",
          "url": "https://github.com/equinor/ert/commit/2f81dd1cfd19ee4e89d36817a4de195aa85bbd59"
        },
        "date": 1736252302335,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2131199328051197,
            "unit": "iter/sec",
            "range": "stddev: 0.02297207905254267",
            "extra": "mean: 4.6921936716 sec\nrounds: 5"
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
          "id": "00b015fcf79da832e5cd8e400cd9d95c58561529",
          "message": "Remove job validation from everest config code",
          "timestamp": "2025-01-07T13:34:09+01:00",
          "tree_id": "8e17a8b3e15557f29deb72cdd04692166a1c1e2a",
          "url": "https://github.com/equinor/ert/commit/00b015fcf79da832e5cd8e400cd9d95c58561529"
        },
        "date": 1736253363932,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21903128627635435,
            "unit": "iter/sec",
            "range": "stddev: 0.02351995839858765",
            "extra": "mean: 4.565557811400003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "stephan.dehoop@tno.nl",
            "name": "Stephan de Hoop",
            "username": "StephanDeHoop"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eb9b04a589b3f1d777ccb4c7d4dd31ef42d9d011",
          "message": "Non optional everest model and nonzero realizations (#9577)\n\nFor EverestConfig require Model and realizations.len() > 0",
          "timestamp": "2025-01-07T17:10:38+01:00",
          "tree_id": "a0985d90212d4529c3053de199abcaa11c1fe21d",
          "url": "https://github.com/equinor/ert/commit/eb9b04a589b3f1d777ccb4c7d4dd31ef42d9d011"
        },
        "date": 1736266349702,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21896301576265956,
            "unit": "iter/sec",
            "range": "stddev: 0.017399442138751015",
            "extra": "mean: 4.5669813074000105 sec\nrounds: 5"
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
          "id": "71275280104b46942e41a907a76342e262928aeb",
          "message": "Remove dependency on ropt config in forward model evaluations",
          "timestamp": "2025-01-08T07:09:10+01:00",
          "tree_id": "1e8e94d36d177a58cca738d36f9f67b01bbf69fb",
          "url": "https://github.com/equinor/ert/commit/71275280104b46942e41a907a76342e262928aeb"
        },
        "date": 1736316665675,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21961637110103988,
            "unit": "iter/sec",
            "range": "stddev: 0.03879839145311598",
            "extra": "mean: 4.553394607999991 sec\nrounds: 5"
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
          "id": "383d6454c982da42a08c1f864fa85aa4c19f21e1",
          "message": "Use resfo for snake_oil_simulator",
          "timestamp": "2025-01-08T09:03:41+01:00",
          "tree_id": "75087374f7c1d0cad7d051167a04c21a69ba62ce",
          "url": "https://github.com/equinor/ert/commit/383d6454c982da42a08c1f864fa85aa4c19f21e1"
        },
        "date": 1736323538773,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2137166107562481,
            "unit": "iter/sec",
            "range": "stddev: 0.011818798830819769",
            "extra": "mean: 4.679093480199993 sec\nrounds: 5"
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
          "id": "e32c0adbff11fb48db0394bfee2ae7111ed6869f",
          "message": "Make timestamps in dmesg output human readable",
          "timestamp": "2025-01-09T09:00:10+01:00",
          "tree_id": "e87d8cf704d33adad00ded5775b25c0ea32842e6",
          "url": "https://github.com/equinor/ert/commit/e32c0adbff11fb48db0394bfee2ae7111ed6869f"
        },
        "date": 1736409721157,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21833625860587397,
            "unit": "iter/sec",
            "range": "stddev: 0.028542746552989325",
            "extra": "mean: 4.5800913068 sec\nrounds: 5"
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
          "id": "40d33bde4b53393573e455227970f7c6868e0951",
          "message": "Explicitly sort dicts in test_api_snapshots",
          "timestamp": "2025-01-09T09:05:49+01:00",
          "tree_id": "fe50da4804b62dc4ad6b15962c81335cf03d2423",
          "url": "https://github.com/equinor/ert/commit/40d33bde4b53393573e455227970f7c6868e0951"
        },
        "date": 1736410058124,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21783493169285315,
            "unit": "iter/sec",
            "range": "stddev: 0.022408783739957087",
            "extra": "mean: 4.5906319626000025 sec\nrounds: 5"
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
          "id": "2a3c5b60137d6879896ee30552c2e2e6ed36d02a",
          "message": "Remove test_run_egg_model\n\ntest_egg_snapshot does close to the same thing",
          "timestamp": "2025-01-09T09:06:40+01:00",
          "tree_id": "ae5ea5e9ff68405e9e59d9f0dbca57082a5669ad",
          "url": "https://github.com/equinor/ert/commit/2a3c5b60137d6879896ee30552c2e2e6ed36d02a"
        },
        "date": 1736410113159,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2200678854876305,
            "unit": "iter/sec",
            "range": "stddev: 0.01888290737861992",
            "extra": "mean: 4.5440523853999935 sec\nrounds: 5"
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
          "id": "71e38f20fdec8d89f4c63348ec2ec766fd279119",
          "message": "Make realization number span attribute",
          "timestamp": "2025-01-09T09:23:47+01:00",
          "tree_id": "dd91b82eebad34d4f7bf8822b74f9b0033ccc11f",
          "url": "https://github.com/equinor/ert/commit/71e38f20fdec8d89f4c63348ec2ec766fd279119"
        },
        "date": 1736411138484,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21404064008280457,
            "unit": "iter/sec",
            "range": "stddev: 0.05912433343272758",
            "extra": "mean: 4.672009949200003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49289030+HakonSohoel@users.noreply.github.com",
            "name": "Håkon Steinkopf Søhoel",
            "username": "HakonSohoel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3a31ffb7788282a00ff39bc8e99016e1fbc3e602",
          "message": "Include spans for dark storage (#9535)",
          "timestamp": "2025-01-09T09:50:41+01:00",
          "tree_id": "8982d9e3891ad3550a6322426dcd9954d9694245",
          "url": "https://github.com/equinor/ert/commit/3a31ffb7788282a00ff39bc8e99016e1fbc3e602"
        },
        "date": 1736412758356,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21784002562900132,
            "unit": "iter/sec",
            "range": "stddev: 0.08374957890094412",
            "extra": "mean: 4.5905246159999935 sec\nrounds: 5"
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
          "id": "c4217b16d52c6cac0a6227cf314d7c68581c9523",
          "message": "Remove exec_env\n\nThis feature has been deprecated since 2019 and can finally be removed. Its last user\nwas the RMS forward model step, which no longer uses it.",
          "timestamp": "2025-01-09T11:24:28+01:00",
          "tree_id": "d33b5c86ee9ec64ca9b14e0e991d404ea957586c",
          "url": "https://github.com/equinor/ert/commit/c4217b16d52c6cac0a6227cf314d7c68581c9523"
        },
        "date": 1736418380398,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21847536580495114,
            "unit": "iter/sec",
            "range": "stddev: 0.027137010108530375",
            "extra": "mean: 4.577175080199993 sec\nrounds: 5"
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
          "id": "3b745a6e027400ec011b2e75a55078b1f9140acf",
          "message": "Fix typo in dependencies",
          "timestamp": "2025-01-09T14:04:46+01:00",
          "tree_id": "4217a2b591beffabe20d91a8b8efaaacf011b49f",
          "url": "https://github.com/equinor/ert/commit/3b745a6e027400ec011b2e75a55078b1f9140acf"
        },
        "date": 1736428047295,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22026616786278097,
            "unit": "iter/sec",
            "range": "stddev: 0.012002047937071825",
            "extra": "mean: 4.539961854800003 sec\nrounds: 5"
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
      }
    ]
  }
}