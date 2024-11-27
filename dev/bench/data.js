window.BENCHMARK_DATA = {
  "lastUpdate": 1732696934294,
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
          "id": "20e9fff2409a4a351887f45e0155dfb2e4208bd7",
          "message": "Remove unused files from test-data/open_shut_state_modifier",
          "timestamp": "2024-11-21T14:42:43+01:00",
          "tree_id": "8ad96d8afe86f8a0934bdc6237f2f3660606d88a",
          "url": "https://github.com/equinor/ert/commit/20e9fff2409a4a351887f45e0155dfb2e4208bd7"
        },
        "date": 1732196674778,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19354800105716952,
            "unit": "iter/sec",
            "range": "stddev: 0.03929361310869673",
            "extra": "mean: 5.166676971800001 sec\nrounds: 5"
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
          "id": "430a782d8fde5833ccffb927fbd332a94dc7d0e5",
          "message": "Remove unused files from test-data/eclipse",
          "timestamp": "2024-11-21T14:44:04+01:00",
          "tree_id": "a0ee072205f61bed5c84b72c51ebec6db84b9fc9",
          "url": "https://github.com/equinor/ert/commit/430a782d8fde5833ccffb927fbd332a94dc7d0e5"
        },
        "date": 1732196754364,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1919791632094739,
            "unit": "iter/sec",
            "range": "stddev: 0.02833289490247708",
            "extra": "mean: 5.2088986288 sec\nrounds: 5"
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
          "id": "adda9e04c1d2851e660169ef9a376049eea5d550",
          "message": "Replace Flask with FastAPI",
          "timestamp": "2024-11-21T14:46:09+01:00",
          "tree_id": "387891bda11b1f13497a62711084025657b72618",
          "url": "https://github.com/equinor/ert/commit/adda9e04c1d2851e660169ef9a376049eea5d550"
        },
        "date": 1732196883703,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19491331697103567,
            "unit": "iter/sec",
            "range": "stddev: 0.018720853496559556",
            "extra": "mean: 5.130485774600004 sec\nrounds: 5"
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
          "id": "8b4dc2a7d100700641ea5c44c54c9ba0c23d2d65",
          "message": "Make find_smry() precise\n\nThis solves a bug where find_unsmry() would not be able to locate\nthe correct summary file even if it has the correct base for the\nEclipse deck",
          "timestamp": "2024-11-22T09:13:20+01:00",
          "tree_id": "bdbdfce8161878af0b6c38dba42059ecb45800a2",
          "url": "https://github.com/equinor/ert/commit/8b4dc2a7d100700641ea5c44c54c9ba0c23d2d65"
        },
        "date": 1732263450247,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19198470497849743,
            "unit": "iter/sec",
            "range": "stddev: 0.03420532938387256",
            "extra": "mean: 5.208748270399985 sec\nrounds: 5"
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
          "id": "52b1e2c274fc2839cd36345431d4eef6d06fbc33",
          "message": "Fix tests writing files to the source tree",
          "timestamp": "2024-11-22T10:02:07+01:00",
          "tree_id": "e5e165b82f21e98522fa79e077915bbbd3bcd16c",
          "url": "https://github.com/equinor/ert/commit/52b1e2c274fc2839cd36345431d4eef6d06fbc33"
        },
        "date": 1732266244625,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19213493756311287,
            "unit": "iter/sec",
            "range": "stddev: 0.03003362364688835",
            "extra": "mean: 5.204675488400011 sec\nrounds: 5"
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
          "id": "e0d923b0af2a15a7a3c55f6e9179b0d4f202f166",
          "message": "fixup get_ensemble_responses logic in dark storage",
          "timestamp": "2024-11-22T11:30:38+01:00",
          "tree_id": "c29d349e0bdb2fa70b08592763f8f1ef10091877",
          "url": "https://github.com/equinor/ert/commit/e0d923b0af2a15a7a3c55f6e9179b0d4f202f166"
        },
        "date": 1732271553834,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19467722427034373,
            "unit": "iter/sec",
            "range": "stddev: 0.02816589297512203",
            "extra": "mean: 5.136707715799991 sec\nrounds: 5"
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
          "id": "968a97c8babf1e755ccab74df358eca778377e84",
          "message": "Align has_data with empty response config .keys",
          "timestamp": "2024-11-22T12:40:09+01:00",
          "tree_id": "89c4293bcfab02856a0b26b3fb2e2996f0c85fc4",
          "url": "https://github.com/equinor/ert/commit/968a97c8babf1e755ccab74df358eca778377e84"
        },
        "date": 1732275721739,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18827087648756913,
            "unit": "iter/sec",
            "range": "stddev: 0.02621250911694178",
            "extra": "mean: 5.311495960800005 sec\nrounds: 5"
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
          "id": "edc56e1fc70ea098d3c1948ad62d21642a9f7dbf",
          "message": "Clean up everest tests",
          "timestamp": "2024-11-25T08:01:26+01:00",
          "tree_id": "5d3f19ef64eee0ad1276f3cd19f74f71f0483f78",
          "url": "https://github.com/equinor/ert/commit/edc56e1fc70ea098d3c1948ad62d21642a9f7dbf"
        },
        "date": 1732518192042,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1970148106735814,
            "unit": "iter/sec",
            "range": "stddev: 0.02709782905823896",
            "extra": "mean: 5.0757605308000056 sec\nrounds: 5"
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
          "id": "306f821714c2cc7185da25f5438ca93cb85d76d3",
          "message": "Replace jobs dir with a symlink",
          "timestamp": "2024-11-25T08:02:24+01:00",
          "tree_id": "ff9aea94ef44504282919cae68bacd7a3b1c2946",
          "url": "https://github.com/equinor/ert/commit/306f821714c2cc7185da25f5438ca93cb85d76d3"
        },
        "date": 1732518252425,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1911202641427395,
            "unit": "iter/sec",
            "range": "stddev: 0.035086335938649585",
            "extra": "mean: 5.232307544599996 sec\nrounds: 5"
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
          "id": "7acec5cae10b2c4774839eac486e0f7bb95cd910",
          "message": "Remove unused files from test-data/valid_config_file",
          "timestamp": "2024-11-25T08:02:59+01:00",
          "tree_id": "12d4352f8e83115758208e60d5fbc4b754704985",
          "url": "https://github.com/equinor/ert/commit/7acec5cae10b2c4774839eac486e0f7bb95cd910"
        },
        "date": 1732518294452,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1924180816360404,
            "unit": "iter/sec",
            "range": "stddev: 0.03236805996069496",
            "extra": "mean: 5.197016784999988 sec\nrounds: 5"
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
          "id": "233bee235736095eff9e97567622e60e55da20a3",
          "message": "Add memory test for some plotapi endpoints",
          "timestamp": "2024-11-25T09:15:13+01:00",
          "tree_id": "498b2b1431e8ba8f6fea597a9670bc0d1a1ce266",
          "url": "https://github.com/equinor/ert/commit/233bee235736095eff9e97567622e60e55da20a3"
        },
        "date": 1732522623991,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1922964332479814,
            "unit": "iter/sec",
            "range": "stddev: 0.045695956206649124",
            "extra": "mean: 5.2003044628 sec\nrounds: 5"
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
          "id": "0a4bad63a11f0f9d6ea1671415ae55383a64c87e",
          "message": "Remove --iter-num from ert cli (#9276)\n\nThis option is no longer available\r\nand is ignored by the application",
          "timestamp": "2024-11-25T09:16:39+01:00",
          "tree_id": "b38ac7bd4324447cb3c714bc3c24aed70254dd8d",
          "url": "https://github.com/equinor/ert/commit/0a4bad63a11f0f9d6ea1671415ae55383a64c87e"
        },
        "date": 1732522714364,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18962108509115977,
            "unit": "iter/sec",
            "range": "stddev: 0.03051962987155439",
            "extra": "mean: 5.2736751270000015 sec\nrounds: 5"
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
          "id": "bd0538e9b794a8083b2221f8b0c50bb2b2608151",
          "message": "Adjust memory test for unstable/high peak usage\n\nhttps://github.com/equinor/ert/actions/runs/12006134015/job/33464079712?pr=9047\n`FAILED tests/ert/unit_tests/gui/tools/plot/test_plot_api.py::test_plot_api_big_summary_memory_usage[1000-100-100-950] - assert 1301.3557224273682 < 950\n`",
          "timestamp": "2024-11-25T10:38:32+01:00",
          "tree_id": "e4caccda1035c548b0d8728069ceb2fb815647d6",
          "url": "https://github.com/equinor/ert/commit/bd0538e9b794a8083b2221f8b0c50bb2b2608151"
        },
        "date": 1732527628558,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18945509745135816,
            "unit": "iter/sec",
            "range": "stddev: 0.015067811929970186",
            "extra": "mean: 5.278295561600004 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "augustommg@gmail.com",
            "name": "AugustoMagalhaes",
            "username": "AugustoMagalhaes"
          },
          "committer": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "3a8ffd836cdbc65fa99523e1b3286051077d7f74",
          "message": "Adds teardown to avoid flaky tests with run_dialog",
          "timestamp": "2024-11-25T10:50:12+01:00",
          "tree_id": "f66ea5e49082bb599d2e173dfd345da9c1882f53",
          "url": "https://github.com/equinor/ert/commit/3a8ffd836cdbc65fa99523e1b3286051077d7f74"
        },
        "date": 1732528323270,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19456414570711983,
            "unit": "iter/sec",
            "range": "stddev: 0.030891273185742755",
            "extra": "mean: 5.139693114399989 sec\nrounds: 5"
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
          "id": "fe9926bd34c7298c0ace0c4b73774cda935fa734",
          "message": "Cache ensemble_state",
          "timestamp": "2024-11-25T13:41:37+01:00",
          "tree_id": "40aba6e79af3ce1084741be3dfa0249af639eb2d",
          "url": "https://github.com/equinor/ert/commit/fe9926bd34c7298c0ace0c4b73774cda935fa734"
        },
        "date": 1732538611765,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1919068643083932,
            "unit": "iter/sec",
            "range": "stddev: 0.05743047757617601",
            "extra": "mean: 5.210861026799989 sec\nrounds: 5"
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
          "id": "e33dfe75f022372c6195da861ddba37f146519a9",
          "message": "Upgrade pre-commit ruff 0.7.3 -> 0.8.0",
          "timestamp": "2024-11-25T15:06:40+01:00",
          "tree_id": "7e4b8ed5f116b9ea8718977a9183535ec68376b7",
          "url": "https://github.com/equinor/ert/commit/e33dfe75f022372c6195da861ddba37f146519a9"
        },
        "date": 1732543711387,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19396551495225098,
            "unit": "iter/sec",
            "range": "stddev: 0.028487240631395443",
            "extra": "mean: 5.155555616400022 sec\nrounds: 5"
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
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "4485f41bc30417fc3d4dd36bd5d43ce62099c373",
          "message": "Mark a test as flaky\n\nThis is a very rare occurence, reruns should be sufficient",
          "timestamp": "2024-11-25T15:24:58+01:00",
          "tree_id": "5a632a6a172c30eeae62f608cabe96b6c9226881",
          "url": "https://github.com/equinor/ert/commit/4485f41bc30417fc3d4dd36bd5d43ce62099c373"
        },
        "date": 1732544805756,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18920632357885153,
            "unit": "iter/sec",
            "range": "stddev: 0.11250853964733565",
            "extra": "mean: 5.2852356151999915 sec\nrounds: 5"
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
          "id": "6a4bcd2e86a23b9eaea0b8a1510694bdb5aab5b3",
          "message": "Ignore expected warnings in test",
          "timestamp": "2024-11-25T15:33:23+01:00",
          "tree_id": "bcf641507e38dd1b66694da593058f34a9ff391b",
          "url": "https://github.com/equinor/ert/commit/6a4bcd2e86a23b9eaea0b8a1510694bdb5aab5b3"
        },
        "date": 1732545313755,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19432645749479735,
            "unit": "iter/sec",
            "range": "stddev: 0.05835965963870035",
            "extra": "mean: 5.145979672000005 sec\nrounds: 5"
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
          "id": "c0cb78880bca1cdc045935f60f1e157ea60746b2",
          "message": "Remove report_steps\n\nThis was a remnant from when we could not load different summary\nfiles with different report_steps. This is no longer the case, so\nthis is just a complication.",
          "timestamp": "2024-11-25T16:46:14+01:00",
          "tree_id": "9ab87e9eff109b50a44da1ad82093c5c8b88667d",
          "url": "https://github.com/equinor/ert/commit/c0cb78880bca1cdc045935f60f1e157ea60746b2"
        },
        "date": 1732549685308,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19593875553165482,
            "unit": "iter/sec",
            "range": "stddev: 0.021031451449080443",
            "extra": "mean: 5.1036355584000095 sec\nrounds: 5"
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
          "id": "bb0318635e275ed426e597f8c91a63a581bd7565",
          "message": "Remove multi-realization save logic\n\n(testing)",
          "timestamp": "2024-11-26T08:46:16+01:00",
          "tree_id": "83ff05c81684940f2ddf2b1c85f0f065892318fd",
          "url": "https://github.com/equinor/ert/commit/bb0318635e275ed426e597f8c91a63a581bd7565"
        },
        "date": 1732607291835,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19237468187030143,
            "unit": "iter/sec",
            "range": "stddev: 0.05902346462649664",
            "extra": "mean: 5.198189232999994 sec\nrounds: 5"
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
          "id": "4b2c2f58090658fbc3ac85bebc4e48c48359e26d",
          "message": "Remove unused files from test-data/templating",
          "timestamp": "2024-11-26T10:22:33+01:00",
          "tree_id": "7d92cc687e995aa5bbd7a86187c8573f480d82de",
          "url": "https://github.com/equinor/ert/commit/4b2c2f58090658fbc3ac85bebc4e48c48359e26d"
        },
        "date": 1732613070919,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19472945167182465,
            "unit": "iter/sec",
            "range": "stddev: 0.05068456452254919",
            "extra": "mean: 5.135330025400004 sec\nrounds: 5"
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
          "id": "baccfe19b4be2ae90ca8da0ef477c9f0bf060456",
          "message": "Install everest deps when building readthedocs",
          "timestamp": "2024-11-26T12:15:48+01:00",
          "tree_id": "def7e8b8e07d63e37524ec49c0e5e1738fa0bb4b",
          "url": "https://github.com/equinor/ert/commit/baccfe19b4be2ae90ca8da0ef477c9f0bf060456"
        },
        "date": 1732619862701,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1951988419515447,
            "unit": "iter/sec",
            "range": "stddev: 0.029686839000548502",
            "extra": "mean: 5.122981212399997 sec\nrounds: 5"
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
          "id": "67e756dc9e56d73679877b190343600c21d397b5",
          "message": "Fix extra requirements for readthedocs build",
          "timestamp": "2024-11-26T12:40:01+01:00",
          "tree_id": "b3d67e2db5934891c684b168d6874e1f71cbb720",
          "url": "https://github.com/equinor/ert/commit/67e756dc9e56d73679877b190343600c21d397b5"
        },
        "date": 1732621313463,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1941991921705421,
            "unit": "iter/sec",
            "range": "stddev: 0.01528226801816358",
            "extra": "mean: 5.149352007200003 sec\nrounds: 5"
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
          "id": "71105ba7bc2b93473b87b1d36e2c7f8d52affbe2",
          "message": "Remove unused files from test-data/mocked_test_case",
          "timestamp": "2024-11-26T12:44:53+01:00",
          "tree_id": "9276b248a8a696337b683e2a3cce58906054a477",
          "url": "https://github.com/equinor/ert/commit/71105ba7bc2b93473b87b1d36e2c7f8d52affbe2"
        },
        "date": 1732621605497,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1915184988033237,
            "unit": "iter/sec",
            "range": "stddev: 0.035587246495269155",
            "extra": "mean: 5.22142772759999 sec\nrounds: 5"
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
          "id": "dc5ab03d3013b896c85ee095da20a89db375928b",
          "message": "Remove excess minimum realizations validation in baserunmodel\n\nThis commit removes the additional validation `validate_active_realizations_count` which checked that the number of  realizations to run was greater than the minimum required realizations count.\nThis validation is already done in `model_factory::_setup_ensemble_experiment`",
          "timestamp": "2024-11-26T14:12:41+01:00",
          "tree_id": "d248292d4d0450a5c669c1d0527461d91c22dcad",
          "url": "https://github.com/equinor/ert/commit/dc5ab03d3013b896c85ee095da20a89db375928b"
        },
        "date": 1732626874798,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19477501413344553,
            "unit": "iter/sec",
            "range": "stddev: 0.04043459587765664",
            "extra": "mean: 5.134128750800005 sec\nrounds: 5"
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
          "id": "d7829aca719d2016e4884ef38e464a318444f18b",
          "message": "Initialize ert config with plugins when starting webviz ert",
          "timestamp": "2024-11-26T14:30:14+01:00",
          "tree_id": "824133a85bc64e939f99759d96c5bb5ae006c2e7",
          "url": "https://github.com/equinor/ert/commit/d7829aca719d2016e4884ef38e464a318444f18b"
        },
        "date": 1732627926837,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1908070904512115,
            "unit": "iter/sec",
            "range": "stddev: 0.03657527773268285",
            "extra": "mean: 5.240895386199997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca1806e59d1a6a45a18cbc4a3c4ca51af6b4db0b",
          "message": "Make bsub tests faster (#9350)\n\n* Remove needless wait in driver execute\r\n\r\nDo not wait after last possible attempt in driver execute_with_retry\r\n\r\n* Fix naming of max attempts in driver\r\n\r\n* Add max attempt to lsf test\r\n\r\nRemove retries due to stdout missing",
          "timestamp": "2024-11-26T15:21:17+01:00",
          "tree_id": "a2a5066c6e3cf411c3ec7d8cd65fa875de5deaa6",
          "url": "https://github.com/equinor/ert/commit/ca1806e59d1a6a45a18cbc4a3c4ca51af6b4db0b"
        },
        "date": 1732630987772,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19233637591227304,
            "unit": "iter/sec",
            "range": "stddev: 0.04663167774807077",
            "extra": "mean: 5.199224511 sec\nrounds: 5"
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
          "id": "398aa3e77980a329bc9a91e5a540fb2cc99ea03e",
          "message": "Unpin hypothesis",
          "timestamp": "2024-11-27T09:15:00+01:00",
          "tree_id": "c2d72160ec8103e25d449d1fb65a240b51acf709",
          "url": "https://github.com/equinor/ert/commit/398aa3e77980a329bc9a91e5a540fb2cc99ea03e"
        },
        "date": 1732695417284,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1942073480123078,
            "unit": "iter/sec",
            "range": "stddev: 0.035025050932171516",
            "extra": "mean: 5.149135757400001 sec\nrounds: 5"
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
          "id": "dd7925c48926cce52e2e0cadb3994ec11658cf45",
          "message": "Remove the everest load entry point",
          "timestamp": "2024-11-27T09:23:34+01:00",
          "tree_id": "af2ddaa3ce8c3b0378506b91930c215d5ca470b3",
          "url": "https://github.com/equinor/ert/commit/dd7925c48926cce52e2e0cadb3994ec11658cf45"
        },
        "date": 1732695922113,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1922525504311773,
            "unit": "iter/sec",
            "range": "stddev: 0.02507621422219319",
            "extra": "mean: 5.201491464000009 sec\nrounds: 5"
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
          "id": "447d647c224ffc7f94fcd905d4597aff55a766f4",
          "message": "Use explicit everest config components when creating ropt config object",
          "timestamp": "2024-11-27T17:40:26+09:00",
          "tree_id": "eb504b41594721a723b792d04ac07834cf0bf29b",
          "url": "https://github.com/equinor/ert/commit/447d647c224ffc7f94fcd905d4597aff55a766f4"
        },
        "date": 1732696933729,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19115550773424858,
            "unit": "iter/sec",
            "range": "stddev: 0.029059871232901543",
            "extra": "mean: 5.231342857199996 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}