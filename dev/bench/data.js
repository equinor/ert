window.BENCHMARK_DATA = {
  "lastUpdate": 1730798771017,
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
          "id": "d63756369c15bcdf9b14d1f167d7f01320d816c3",
          "message": "Fix typo in variable name in testkomodo.sh",
          "timestamp": "2024-10-29T11:55:48+01:00",
          "tree_id": "9e0005a4a649b5bb2d1b6e7484b0dc61ac0d7341",
          "url": "https://github.com/equinor/ert/commit/d63756369c15bcdf9b14d1f167d7f01320d816c3"
        },
        "date": 1730199456443,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1952473238806674,
            "unit": "iter/sec",
            "range": "stddev: 0.041276735662446135",
            "extra": "mean: 5.121709123200003 sec\nrounds: 5"
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
          "id": "132ca3c741bc8f22f75d5d7cabb58a281150a3d4",
          "message": "Brush up test_ert_config\n\nTest code should not be changed by this commit, this is only cleanup\nlike:\n* String whitespace formatting for ert config files\n* Usage of pathlib\n* Expand some test names\n* Use fixture for changing to tmp_path",
          "timestamp": "2024-10-29T13:27:59+01:00",
          "tree_id": "a6d300e35fdb7981f5f05bcf1eb66eae4189c259",
          "url": "https://github.com/equinor/ert/commit/132ca3c741bc8f22f75d5d7cabb58a281150a3d4"
        },
        "date": 1730204987732,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19161156412727617,
            "unit": "iter/sec",
            "range": "stddev: 0.022541461883642793",
            "extra": "mean: 5.2188916914000005 sec\nrounds: 5"
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
          "id": "d329f1c446df7bb1345cafd33ee23134e552bf19",
          "message": "Have run simulation button disabled when running",
          "timestamp": "2024-10-29T13:42:00+01:00",
          "tree_id": "59991235bbb691166758394ed26c7feb8a2fc7f5",
          "url": "https://github.com/equinor/ert/commit/d329f1c446df7bb1345cafd33ee23134e552bf19"
        },
        "date": 1730205825532,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19379035865461405,
            "unit": "iter/sec",
            "range": "stddev: 0.02587151582979508",
            "extra": "mean: 5.160215435600003 sec\nrounds: 5"
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
          "id": "e397968a49c1d71fcd6cd6d8576bfa844265b9ab",
          "message": "Make test fixture minimum_case really minimal",
          "timestamp": "2024-10-29T14:25:46+01:00",
          "tree_id": "d66b372d6ff87a46e1ea37d10b85d15cddac7ed3",
          "url": "https://github.com/equinor/ert/commit/e397968a49c1d71fcd6cd6d8576bfa844265b9ab"
        },
        "date": 1730208453777,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19237271209429074,
            "unit": "iter/sec",
            "range": "stddev: 0.11566210336350013",
            "extra": "mean: 5.198242459200003 sec\nrounds: 5"
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
          "id": "0aa55b414ca0e121f357a53590c0582148131813",
          "message": "Detect dark_mode in sidepanel",
          "timestamp": "2024-10-29T15:18:14+01:00",
          "tree_id": "36342805c14fdd5d593bb272e3d7047deb616710",
          "url": "https://github.com/equinor/ert/commit/0aa55b414ca0e121f357a53590c0582148131813"
        },
        "date": 1730211610478,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19082612671494306,
            "unit": "iter/sec",
            "range": "stddev: 0.022965478674225173",
            "extra": "mean: 5.240372569599993 sec\nrounds: 5"
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
          "id": "28763a6551f19eadb9cc4675baa9d2f405f10921",
          "message": "Increment storage version\n\nPut polars migration into separate version\r\ndue to 2024.10 ERT being still xarray",
          "timestamp": "2024-10-29T15:39:24+01:00",
          "tree_id": "07a9b8692b3231f1ae013ae1797bc907651db3a6",
          "url": "https://github.com/equinor/ert/commit/28763a6551f19eadb9cc4675baa9d2f405f10921"
        },
        "date": 1730212875866,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18802466468712978,
            "unit": "iter/sec",
            "range": "stddev: 0.0342248591953596",
            "extra": "mean: 5.318451181200004 sec\nrounds: 5"
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
          "id": "f9347350ae9c409d19872101e1a6479a20ed9eea",
          "message": "Show obs counts correctly summary panel",
          "timestamp": "2024-10-30T13:20:14+01:00",
          "tree_id": "b510a8ab6abf4c337865c195d3e43363bf32d9ca",
          "url": "https://github.com/equinor/ert/commit/f9347350ae9c409d19872101e1a6479a20ed9eea"
        },
        "date": 1730290928464,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1938469715755869,
            "unit": "iter/sec",
            "range": "stddev: 0.02412473537954339",
            "extra": "mean: 5.158708396999998 sec\nrounds: 5"
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
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "98576cc8ad12e751a19340e8fa350ef08ed3ee59",
          "message": "Increase sleep in memory profile test from 0.1 -> 0.15",
          "timestamp": "2024-10-31T08:31:09+01:00",
          "tree_id": "4c6ccb8737b71f6d7b3377f7515303b6d5b27f26",
          "url": "https://github.com/equinor/ert/commit/98576cc8ad12e751a19340e8fa350ef08ed3ee59"
        },
        "date": 1730359979652,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19419065877406683,
            "unit": "iter/sec",
            "range": "stddev: 0.041257105184121186",
            "extra": "mean: 5.149578287200006 sec\nrounds: 5"
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
          "id": "c4e136546fa7f03090110e8dac1647c95e0f58b8",
          "message": "Log runtime for individual forward model steps\n\nThis is meant for making statistics pr forward model step name.\n\nWhile testing on poly_example and local queue, there\nis some odd occurences of the logged message that start_time\nis None, this is not believed to happen often in production.",
          "timestamp": "2024-10-31T09:26:43+01:00",
          "tree_id": "d4651fd3a80f2f0fe8fa630d1418271291a22cb0",
          "url": "https://github.com/equinor/ert/commit/c4e136546fa7f03090110e8dac1647c95e0f58b8"
        },
        "date": 1730363313927,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19573123415225843,
            "unit": "iter/sec",
            "range": "stddev: 0.013071220862864061",
            "extra": "mean: 5.109046618599995 sec\nrounds: 5"
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
          "id": "40c833805eae51f790d20d3bde1038c40b6a27a2",
          "message": "Test field update using heat equation instead of poly",
          "timestamp": "2024-10-31T11:41:44+01:00",
          "tree_id": "6e6c5e9c3692347b9b511c126d5e02136b700280",
          "url": "https://github.com/equinor/ert/commit/40c833805eae51f790d20d3bde1038c40b6a27a2"
        },
        "date": 1730371414413,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19560114247813806,
            "unit": "iter/sec",
            "range": "stddev: 0.020794376431325683",
            "extra": "mean: 5.112444576399997 sec\nrounds: 5"
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
          "id": "3eb3e3cbd65068b465e38cf0a41b5d54693743dd",
          "message": "Add readthedocs for everest",
          "timestamp": "2024-10-31T12:02:37+01:00",
          "tree_id": "5eea284ebe8496ca09c6f09eaee0686c9366bc9f",
          "url": "https://github.com/equinor/ert/commit/3eb3e3cbd65068b465e38cf0a41b5d54693743dd"
        },
        "date": 1730372672197,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19206740369335817,
            "unit": "iter/sec",
            "range": "stddev: 0.021636133729286657",
            "extra": "mean: 5.206505532799997 sec\nrounds: 5"
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
          "id": "aac8f804ffc6ffa25bb7127ade296f2f0aac3952",
          "message": "Remove function only called in test",
          "timestamp": "2024-10-31T12:03:18+01:00",
          "tree_id": "0021ef19fea09eaac61b6b4e7406d6db63efa941",
          "url": "https://github.com/equinor/ert/commit/aac8f804ffc6ffa25bb7127ade296f2f0aac3952"
        },
        "date": 1730372707382,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19671715994098707,
            "unit": "iter/sec",
            "range": "stddev: 0.0388839294678522",
            "extra": "mean: 5.0834406124 sec\nrounds: 5"
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
          "id": "ca68c84724fc67d44779725544014ab836ee1c4c",
          "message": "Remove iEverest",
          "timestamp": "2024-10-31T12:06:00+01:00",
          "tree_id": "e3f63400b7a087c0a892a3d1c8bd655c3224837c",
          "url": "https://github.com/equinor/ert/commit/ca68c84724fc67d44779725544014ab836ee1c4c"
        },
        "date": 1730372876680,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19203659249219796,
            "unit": "iter/sec",
            "range": "stddev: 0.0364210809898341",
            "extra": "mean: 5.207340887599992 sec\nrounds: 5"
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
          "id": "70e85a76a6a27f5716c45c1ec4b8b276daf4f808",
          "message": "Fix unbound event\n\nValidation is raised from dispatch_event_from_json so\nevent is unbound.",
          "timestamp": "2024-10-31T12:55:38+01:00",
          "tree_id": "0fa6ff1f7a01f542801a69e27d934f2cf9913ef8",
          "url": "https://github.com/equinor/ert/commit/70e85a76a6a27f5716c45c1ec4b8b276daf4f808"
        },
        "date": 1730375846438,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1991651433045672,
            "unit": "iter/sec",
            "range": "stddev: 0.013230070044671145",
            "extra": "mean: 5.020958905800001 sec\nrounds: 5"
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
          "id": "d29493050df0077131058fe80d04d9b2576944fe",
          "message": "Change button text `Debug Info`->`Copy Debug Info`\n\nThe button text was misleading, as it could be interpreted as \"show debug info\" instead of it copying the debug info to the clipboard.",
          "timestamp": "2024-10-31T14:52:16+01:00",
          "tree_id": "11e229cf8aa2273fe817ee6bb8bbcd2662d3547a",
          "url": "https://github.com/equinor/ert/commit/d29493050df0077131058fe80d04d9b2576944fe"
        },
        "date": 1730382843163,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19291091671620175,
            "unit": "iter/sec",
            "range": "stddev: 0.0268233687214356",
            "extra": "mean: 5.183739816400004 sec\nrounds: 5"
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
          "id": "c077aad7c74ce1d48e26dc1703545263a4d6f152",
          "message": "Make plot_api return all observations for response",
          "timestamp": "2024-10-31T14:39:37Z",
          "tree_id": "e633ef272ba93fc496b2ddc23ad75338c0221864",
          "url": "https://github.com/equinor/ert/commit/c077aad7c74ce1d48e26dc1703545263a4d6f152"
        },
        "date": 1730385691493,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19462632084366618,
            "unit": "iter/sec",
            "range": "stddev: 0.01928909080231142",
            "extra": "mean: 5.138051193000001 sec\nrounds: 5"
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
          "id": "8c312f2c2552203f62156054223c160f613bdd83",
          "message": "Fix logging of step runtime statistics\n\nThis had to be moved to _ensemble.py as the state inside _snapshot.py\nonly had information about steps that fell into the same batching\nwindow, thus long lasting steps would have start_time=None",
          "timestamp": "2024-11-01T07:44:34+01:00",
          "tree_id": "62f20f1cd653ea7548f2c3321ab7f63a7dd62ef4",
          "url": "https://github.com/equinor/ert/commit/8c312f2c2552203f62156054223c160f613bdd83"
        },
        "date": 1730443586578,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.193994606010088,
            "unit": "iter/sec",
            "range": "stddev: 0.03415446621832291",
            "extra": "mean: 5.154782499199996 sec\nrounds: 5"
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
          "id": "3b1db99eb00cc8c783063eed20c2e53fa13ec991",
          "message": "Have CopyDebugInfoButton change text to `Copied...` on click\n\nThis commit refactors the copydebuginfobutton in run_dialog.py into its own class, and makes it change text `Copy Debug Info\" -> `Copied...` when clicked while running the callback passed to it. After one second, it changes it text back to `Copy Debug Info`.",
          "timestamp": "2024-11-01T09:39:47+01:00",
          "tree_id": "4cc66a61a1f773952099a0ca34dc6a49bc637258",
          "url": "https://github.com/equinor/ert/commit/3b1db99eb00cc8c783063eed20c2e53fa13ec991"
        },
        "date": 1730450500938,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19274783485756736,
            "unit": "iter/sec",
            "range": "stddev: 0.021090395544056846",
            "extra": "mean: 5.188125722599989 sec\nrounds: 5"
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
          "id": "67ac195545097204906b1f9a1dfce531a4a745b0",
          "message": "Update docs path",
          "timestamp": "2024-11-01T10:51:26+01:00",
          "tree_id": "f656a0e906a4acf753e0767cf8b456bfbbfa224e",
          "url": "https://github.com/equinor/ert/commit/67ac195545097204906b1f9a1dfce531a4a745b0"
        },
        "date": 1730454796832,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1935736941597423,
            "unit": "iter/sec",
            "range": "stddev: 0.04787168965994167",
            "extra": "mean: 5.165991197000006 sec\nrounds: 5"
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
          "id": "6454d3a8a8163665b7a39d4d4cc77f867b38b8e5",
          "message": "Skip logging of short fm steps\n\nThere are more short forward model time steps than we want the logging\nsystem to handle.",
          "timestamp": "2024-11-01T10:52:13+01:00",
          "tree_id": "5b6c02efe727ff0f0ad7d7511a14bf8d659b1b76",
          "url": "https://github.com/equinor/ert/commit/6454d3a8a8163665b7a39d4d4cc77f867b38b8e5"
        },
        "date": 1730454849177,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19433471679065184,
            "unit": "iter/sec",
            "range": "stddev: 0.006063808994274784",
            "extra": "mean: 5.145760965999995 sec\nrounds: 5"
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
          "id": "66aa073a0831991022c17dff7a5fd09688298e37",
          "message": "Use maxprocesses=2 for cli tests\n\nCo-authored-by: Eivind Jahren <ejah@equinor.com>",
          "timestamp": "2024-11-01T11:32:36+01:00",
          "tree_id": "2262601ceb4df8b82c20f24d137caf1c9f2e402a",
          "url": "https://github.com/equinor/ert/commit/66aa073a0831991022c17dff7a5fd09688298e37"
        },
        "date": 1730457263680,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19656487553447052,
            "unit": "iter/sec",
            "range": "stddev: 0.03869466312127469",
            "extra": "mean: 5.087378898599996 sec\nrounds: 5"
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
          "id": "044db5aec48bbda9839cf197c7faac5c8e72a058",
          "message": "Revert \"Increase sleep in memory profile test from 0.1 -> 0.15\"\n\nThis reverts commit 98576cc8ad12e751a19340e8fa350ef08ed3ee59.\n\nChanging the sleep time affects the rate of memory allocation,\nwhich the assert further down depends on.",
          "timestamp": "2024-11-01T12:39:24+01:00",
          "tree_id": "b13dbc3fab2b8f853aa8b692b223d51e71a795bc",
          "url": "https://github.com/equinor/ert/commit/044db5aec48bbda9839cf197c7faac5c8e72a058"
        },
        "date": 1730461281432,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19200743007680682,
            "unit": "iter/sec",
            "range": "stddev: 0.04340572702924938",
            "extra": "mean: 5.2081317873999975 sec\nrounds: 5"
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
          "id": "7ee89c18b36844c0e4c3ba0f886ecccdc1d7e552",
          "message": "Mitigate flakiness in memory profiling\n\nAnd add some explanation for further debugging",
          "timestamp": "2024-11-01T14:21:48+01:00",
          "tree_id": "99e7dbc9660f1203352392ac251f9504a719f21a",
          "url": "https://github.com/equinor/ert/commit/7ee89c18b36844c0e4c3ba0f886ecccdc1d7e552"
        },
        "date": 1730467418469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19083839842672864,
            "unit": "iter/sec",
            "range": "stddev: 0.034027747519319154",
            "extra": "mean: 5.2400355916000025 sec\nrounds: 5"
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
          "id": "c34060ca08b824c1e5dbedd21cca552dff98b796",
          "message": "Improve logging with open telemetry traces (#9083)\n\nAdd a span processor through the add_span_processor pluggin hook\r\nto export trace information to e.g. azure\r\n---------\r\n\r\nCo-authored-by: Andreas Eknes Lie <andrli@equinor.com>",
          "timestamp": "2024-11-01T14:25:59+01:00",
          "tree_id": "a4b2c2ab5e5967b8505c052d41534d985e8bde95",
          "url": "https://github.com/equinor/ert/commit/c34060ca08b824c1e5dbedd21cca552dff98b796"
        },
        "date": 1730467673141,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1939554861378494,
            "unit": "iter/sec",
            "range": "stddev: 0.02633452409202933",
            "extra": "mean: 5.1558221935999935 sec\nrounds: 5"
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
          "id": "87136c7cbf2f966dced604e36a2dde82a129285d",
          "message": "Add timeout for server",
          "timestamp": "2024-11-01T14:59:14+01:00",
          "tree_id": "095dc0a0f2a245530d269197313acb2f998ecec2",
          "url": "https://github.com/equinor/ert/commit/87136c7cbf2f966dced604e36a2dde82a129285d"
        },
        "date": 1730469665407,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1898364121527562,
            "unit": "iter/sec",
            "range": "stddev: 0.03155378385903209",
            "extra": "mean: 5.267693318999977 sec\nrounds: 5"
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
          "id": "1488d88244f4b2bb2cd165a7197fd0846c1b701a",
          "message": "Remove ert_config from batch sim",
          "timestamp": "2024-11-01T15:03:13+01:00",
          "tree_id": "0806b0089def2c042be2389c9d12d06ee60aaae7",
          "url": "https://github.com/equinor/ert/commit/1488d88244f4b2bb2cd165a7197fd0846c1b701a"
        },
        "date": 1730469902449,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1926733180812877,
            "unit": "iter/sec",
            "range": "stddev: 0.017998264994396853",
            "extra": "mean: 5.190132240199995 sec\nrounds: 5"
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
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "90d11ec9604f2d3b4b2a12cd90ba19ec58fbf1ca",
          "message": "Make scheduler execute yield during spawning of realizations\n\nStarting the realizations in scheduler was blocking all other async tasks\nfrom running. Nothing could connect to ensemble evaluator during this.\nUnder heavy load this could cause Monitor to time out and fail. Now we will\nsleep(0) between each time we create a new subprocess. This will allow\nother asyncio tasks to run.",
          "timestamp": "2024-11-01T15:15:27+01:00",
          "tree_id": "5fd69702e1f912ba4c614ab240104c1f7b93c28b",
          "url": "https://github.com/equinor/ert/commit/90d11ec9604f2d3b4b2a12cd90ba19ec58fbf1ca"
        },
        "date": 1730470638886,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19423774875696875,
            "unit": "iter/sec",
            "range": "stddev: 0.035257273320927464",
            "extra": "mean: 5.148329850400012 sec\nrounds: 5"
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
          "id": "8c679c1f3753359fdf656db36faa5721e182a865",
          "message": "Update ropt dependency to 0.9",
          "timestamp": "2024-11-04T11:10:22+01:00",
          "tree_id": "213486ade147c9856172d5a8c2d4b11b20381248",
          "url": "https://github.com/equinor/ert/commit/8c679c1f3753359fdf656db36faa5721e182a865"
        },
        "date": 1730715132812,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19240231102079516,
            "unit": "iter/sec",
            "range": "stddev: 0.028166166099757847",
            "extra": "mean: 5.1974427682000055 sec\nrounds: 5"
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
          "id": "05dfe1711c20226ee73ff22eb3794272b6b794e7",
          "message": "Create driver from QueueOptions instead of QueueConfig",
          "timestamp": "2024-11-05T10:21:14+01:00",
          "tree_id": "a2240fbc1f1acd9b98b2dff2fd72c14acf3c9bc6",
          "url": "https://github.com/equinor/ert/commit/05dfe1711c20226ee73ff22eb3794272b6b794e7"
        },
        "date": 1730798583967,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19657331009998202,
            "unit": "iter/sec",
            "range": "stddev: 0.04104420423443821",
            "extra": "mean: 5.0871606094000015 sec\nrounds: 5"
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
          "id": "e310860625e56f8daad713991ef56f35ef04d400",
          "message": "Transpose before calculating covariance in test\n\nnp.cov expects rows to be parameters and columns to be realizations",
          "timestamp": "2024-11-05T10:24:16+01:00",
          "tree_id": "01c0b3041f0abd978231c71184edbf4749b148c9",
          "url": "https://github.com/equinor/ert/commit/e310860625e56f8daad713991ef56f35ef04d400"
        },
        "date": 1730798770057,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19046176259498293,
            "unit": "iter/sec",
            "range": "stddev: 0.07114044650732476",
            "extra": "mean: 5.250397698600011 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}