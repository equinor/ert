window.BENCHMARK_DATA = {
  "lastUpdate": 1736774333547,
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
      }
    ]
  }
}