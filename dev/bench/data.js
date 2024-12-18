window.BENCHMARK_DATA = {
  "lastUpdate": 1734526299255,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "d875982c2c94fd55936137990e1053aa38851295",
          "message": "Remove deprecated slurm options `MEMORY` and `MEMORY_PER_CPU`",
          "timestamp": "2024-12-12T12:53:24+01:00",
          "tree_id": "f923ec92785ef76ff9cfc780d1e42f553937bbcc",
          "url": "https://github.com/equinor/ert/commit/d875982c2c94fd55936137990e1053aa38851295"
        },
        "date": 1734004515556,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2179409032641027,
            "unit": "iter/sec",
            "range": "stddev: 0.031233707796770253",
            "extra": "mean: 4.588399813999996 sec\nrounds: 5"
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
          "id": "d27c2d18f15f4e37d36d6da35d0ef8c963807541",
          "message": "Set persistent cells in update table cells",
          "timestamp": "2024-12-12T18:59:55+01:00",
          "tree_id": "46eecaa5995baff79d1604024ea7220dda62c208",
          "url": "https://github.com/equinor/ert/commit/d27c2d18f15f4e37d36d6da35d0ef8c963807541"
        },
        "date": 1734026506014,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21793326549223063,
            "unit": "iter/sec",
            "range": "stddev: 0.01955495229463724",
            "extra": "mean: 4.588560620799996 sec\nrounds: 5"
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
          "id": "4301b63377d4b73678d8ed98669396562e14c975",
          "message": "Add run experiment with design matrix to ensemble experiment panel\n\n- Catagorical data is not treated properly yet, wherein the design\nmatrix group that contains catagorical data will automatically store all\nparameters inside this group to objects; ie, strings.\n- Prefil active realization box with realizations from design matrix\n- Use design_matrix parameters in ensemble experiment\n- add test run cli with design matrix and poly example\n- add test that save parameters internalize DataFrame parameters in the storage\n- add merge function to merge design parameters with existing parameters\n -- Raise Validation error when having multiple overlapping groups\n- Update writting to parameter.txt with categorical values",
          "timestamp": "2024-12-13T00:26:53+01:00",
          "tree_id": "988124ce975ca5a6c89c3ecf6f79168c8b58f900",
          "url": "https://github.com/equinor/ert/commit/4301b63377d4b73678d8ed98669396562e14c975"
        },
        "date": 1734046129192,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21976842864856433,
            "unit": "iter/sec",
            "range": "stddev: 0.02752415245771421",
            "extra": "mean: 4.550244118999996 sec\nrounds: 5"
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
          "id": "1ed8fc3584691dce4a3470129ff2bfe911bc2ccc",
          "message": "Add logging of event processing time when significant",
          "timestamp": "2024-12-13T10:03:13+01:00",
          "tree_id": "c95d214dd313a9de841a6feb13ecd24908187f4b",
          "url": "https://github.com/equinor/ert/commit/1ed8fc3584691dce4a3470129ff2bfe911bc2ccc"
        },
        "date": 1734080708135,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21815696533415874,
            "unit": "iter/sec",
            "range": "stddev: 0.027873626339338563",
            "extra": "mean: 4.583855475199999 sec\nrounds: 5"
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
          "id": "9725d7d71fdf2bdb915238d6498742fb2e54d2f8",
          "message": "Type code in src/ert/resources\n\nSkipping eclrun/flow as it will be dealt with in another PR.",
          "timestamp": "2024-12-13T12:41:17+01:00",
          "tree_id": "a581c3b3924b4fda04f898ed449098e0c948b5fc",
          "url": "https://github.com/equinor/ert/commit/9725d7d71fdf2bdb915238d6498742fb2e54d2f8"
        },
        "date": 1734090194252,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21484107953209544,
            "unit": "iter/sec",
            "range": "stddev: 0.037694267637426934",
            "extra": "mean: 4.654603310399994 sec\nrounds: 5"
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
          "id": "0ca408f16c0e252b104a1ae4228702442edab0f1",
          "message": "Fix ruff issues\n\nThese issues sneaked into main via a rebase while ruff was upgraded",
          "timestamp": "2024-12-13T13:01:39+01:00",
          "tree_id": "e7defd4396166d1025a6ae4bb99e84c928dda4c9",
          "url": "https://github.com/equinor/ert/commit/0ca408f16c0e252b104a1ae4228702442edab0f1"
        },
        "date": 1734091417431,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21617873392737874,
            "unit": "iter/sec",
            "range": "stddev: 0.02639143017563111",
            "extra": "mean: 4.625801908599999 sec\nrounds: 5"
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
          "id": "1af592dde9dc250b17b153e928618bdb813855f4",
          "message": "Avoid version specific sed syntax\n\nThe argument to sed prior to this change triggers something in sed version 4.6 or 4.7\n\nAlso solve contamination issues that has crept in.",
          "timestamp": "2024-12-13T13:45:48+01:00",
          "tree_id": "0b68fa71e0d38539ab1df814d07233670bf75096",
          "url": "https://github.com/equinor/ert/commit/1af592dde9dc250b17b153e928618bdb813855f4"
        },
        "date": 1734094057335,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21759708041954798,
            "unit": "iter/sec",
            "range": "stddev: 0.08156905213378346",
            "extra": "mean: 4.595649896000003 sec\nrounds: 5"
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
          "id": "5ae6ece1bfbfe9dff576d6601624474441a88c0d",
          "message": "Add benchmarking with codspeed",
          "timestamp": "2024-12-13T15:01:20+01:00",
          "tree_id": "7f6bff216b05504e95ed7ca720df8ddfcb1d30d1",
          "url": "https://github.com/equinor/ert/commit/5ae6ece1bfbfe9dff576d6601624474441a88c0d"
        },
        "date": 1734098595294,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2157385424052652,
            "unit": "iter/sec",
            "range": "stddev: 0.036735243320427345",
            "extra": "mean: 4.6352403647999925 sec\nrounds: 5"
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
          "id": "e564b57a42a01495234036f2d34b2e104d52a832",
          "message": "Do not cancel ert run in gui test",
          "timestamp": "2024-12-13T15:25:45+01:00",
          "tree_id": "7b269284c9937f27f82377e2a51d92168e6383ad",
          "url": "https://github.com/equinor/ert/commit/e564b57a42a01495234036f2d34b2e104d52a832"
        },
        "date": 1734100053421,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21813876597228984,
            "unit": "iter/sec",
            "range": "stddev: 0.03014409920012147",
            "extra": "mean: 4.584237907199997 sec\nrounds: 5"
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
          "id": "ebe548ea7413351e74f1c9b378210e3ef3392b61",
          "message": "Reduce logging of gui events to file\n\n- Also increases log level for websockets or asyncio",
          "timestamp": "2024-12-13T15:36:57+01:00",
          "tree_id": "83dde9ce170f4caa329704d03af40171f4df2ad1",
          "url": "https://github.com/equinor/ert/commit/ebe548ea7413351e74f1c9b378210e3ef3392b61"
        },
        "date": 1734100725395,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21909196796051808,
            "unit": "iter/sec",
            "range": "stddev: 0.03546044688253611",
            "extra": "mean: 4.564293293399999 sec\nrounds: 5"
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
            "email": "dan.sava42@gmail.com",
            "name": "Dan Sava",
            "username": "DanSava"
          },
          "distinct": true,
          "id": "72babc57267823fd3fe7b8c6906ea1005eec1974",
          "message": "Add torque params to everserver queue config tests",
          "timestamp": "2024-12-16T18:25:34+09:00",
          "tree_id": "5ddf793f117709eb933209e5bec9070d5961c931",
          "url": "https://github.com/equinor/ert/commit/72babc57267823fd3fe7b8c6906ea1005eec1974"
        },
        "date": 1734341247307,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21064330675388496,
            "unit": "iter/sec",
            "range": "stddev: 0.14127235805009516",
            "extra": "mean: 4.747361857399994 sec\nrounds: 5"
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
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "distinct": true,
          "id": "0d10c5d116aa6b294ae9d3361629cf3e9ebe23db",
          "message": "Automatically change sidebar focus when running simulations",
          "timestamp": "2024-12-16T14:07:39+01:00",
          "tree_id": "b52cecd487f337b128e33d91aac68b639002a08a",
          "url": "https://github.com/equinor/ert/commit/0d10c5d116aa6b294ae9d3361629cf3e9ebe23db"
        },
        "date": 1734354573149,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22039706091287284,
            "unit": "iter/sec",
            "range": "stddev: 0.016044370930411698",
            "extra": "mean: 4.537265587199999 sec\nrounds: 5"
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
          "id": "d4e7b72c2eac3ad7ce31f51d585793639f654e6a",
          "message": "Fix everest batch numbering (#9563)",
          "timestamp": "2024-12-16T14:36:49+01:00",
          "tree_id": "48c65cd113c76535ebb2cee0dea75cba345c70a7",
          "url": "https://github.com/equinor/ert/commit/d4e7b72c2eac3ad7ce31f51d585793639f654e6a"
        },
        "date": 1734356323558,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2168985342088425,
            "unit": "iter/sec",
            "range": "stddev: 0.04184824753979805",
            "extra": "mean: 4.61045070519998 sec\nrounds: 5"
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
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "ab58e603d964ce1126ae85b6f396340850877de6",
          "message": "Remove superfluous layer in ert_config",
          "timestamp": "2024-12-16T14:41:30+01:00",
          "tree_id": "5f1657d93d7f1a2359b029ce25d315fafce013e9",
          "url": "https://github.com/equinor/ert/commit/ab58e603d964ce1126ae85b6f396340850877de6"
        },
        "date": 1734356609593,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21745873067969923,
            "unit": "iter/sec",
            "range": "stddev: 0.02440214123462098",
            "extra": "mean: 4.598573701200007 sec\nrounds: 5"
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
          "id": "4550d408674487ff44a1381876f8b14530e703a9",
          "message": "Handle empty ensemble edge case",
          "timestamp": "2024-12-16T14:49:41+01:00",
          "tree_id": "05f1ba95860b95e559b8053ccfae41cb039e8043",
          "url": "https://github.com/equinor/ert/commit/4550d408674487ff44a1381876f8b14530e703a9"
        },
        "date": 1734357097598,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21506659249148377,
            "unit": "iter/sec",
            "range": "stddev: 0.07902852908449726",
            "extra": "mean: 4.649722620399996 sec\nrounds: 5"
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
          "id": "87cbac355e155db79349e0ed9fc3733d7147769a",
          "message": "Remove env variable that was always false",
          "timestamp": "2024-12-16T15:03:18+01:00",
          "tree_id": "bdca649b515e52327fcbf04c701ff12681e67d0c",
          "url": "https://github.com/equinor/ert/commit/87cbac355e155db79349e0ed9fc3733d7147769a"
        },
        "date": 1734357911775,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.215872622549534,
            "unit": "iter/sec",
            "range": "stddev: 0.024628987265804588",
            "extra": "mean: 4.6323613814 sec\nrounds: 5"
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
          "id": "69535ab2e3a8f31394edec4849756f41baf12ae1",
          "message": "Remove unreliable test: test_gui_snapshot",
          "timestamp": "2024-12-16T15:04:39+01:00",
          "tree_id": "7966cc3eda35bf716354ba03c79163e1248f65b6",
          "url": "https://github.com/equinor/ert/commit/69535ab2e3a8f31394edec4849756f41baf12ae1"
        },
        "date": 1734357992826,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21804041852322464,
            "unit": "iter/sec",
            "range": "stddev: 0.022650697789625963",
            "extra": "mean: 4.586305634400003 sec\nrounds: 5"
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
          "id": "d332898b76bb68cbba381ad904c964d2fa006e7a",
          "message": "Stop using cached seba data",
          "timestamp": "2024-12-16T16:31:40+01:00",
          "tree_id": "217e3714154bb80d8c6b870af2fb0d1e2a293fc7",
          "url": "https://github.com/equinor/ert/commit/d332898b76bb68cbba381ad904c964d2fa006e7a"
        },
        "date": 1734363213052,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21930685420641616,
            "unit": "iter/sec",
            "range": "stddev: 0.01009793739296743",
            "extra": "mean: 4.559821003400009 sec\nrounds: 5"
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
          "id": "f96d3ce2b78b89c7d58824fb84b6944c58babac8",
          "message": "Stop using cached seba data",
          "timestamp": "2024-12-16T16:32:37+01:00",
          "tree_id": "44c6b26df876190d11b21f3fd90eeb9c7a168bc5",
          "url": "https://github.com/equinor/ert/commit/f96d3ce2b78b89c7d58824fb84b6944c58babac8"
        },
        "date": 1734363270252,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.22037485545093788,
            "unit": "iter/sec",
            "range": "stddev: 0.03633777651701471",
            "extra": "mean: 4.537722772199982 sec\nrounds: 5"
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
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "0e1901256b18e622c31a8bb5937eb3e0267b77ef",
          "message": "Name job after its executable when not specified",
          "timestamp": "2024-12-16T18:28:13+01:00",
          "tree_id": "00adfcd6681758c4f8729fde0b38784de4c57f5a",
          "url": "https://github.com/equinor/ert/commit/0e1901256b18e622c31a8bb5937eb3e0267b77ef"
        },
        "date": 1734370208277,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21292027902073588,
            "unit": "iter/sec",
            "range": "stddev: 0.03826610314741605",
            "extra": "mean: 4.696593507199998 sec\nrounds: 5"
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
          "id": "b17963f93942bee4648288964e8a0133adfda1c8",
          "message": "Fix key error on empty summary observations in plotter\n\n\r\n\r\nCo-authored-by: Eivind Jahren <ejah@equinor.com>",
          "timestamp": "2024-12-18T08:16:17+01:00",
          "tree_id": "636a98d4862c118a7c0efc0068beb8fb772d400b",
          "url": "https://github.com/equinor/ert/commit/b17963f93942bee4648288964e8a0133adfda1c8"
        },
        "date": 1734506287950,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2148757785156491,
            "unit": "iter/sec",
            "range": "stddev: 0.016913266146428664",
            "extra": "mean: 4.653851666799994 sec\nrounds: 5"
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
          "id": "01156bb8355596d8dcb22570d963091bdca71c74",
          "message": "Add specific exceptions to bad-dunder-names",
          "timestamp": "2024-12-18T09:04:48+01:00",
          "tree_id": "b5a5857550018617a9288eed2e4d5d33898fe280",
          "url": "https://github.com/equinor/ert/commit/01156bb8355596d8dcb22570d963091bdca71c74"
        },
        "date": 1734509233256,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.21463586830074877,
            "unit": "iter/sec",
            "range": "stddev: 0.010339623610527466",
            "extra": "mean: 4.6590535306 sec\nrounds: 5"
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
          "id": "a6649a18d307b353de1e01428b189a6b335108e6",
          "message": "Unpin pydantic",
          "timestamp": "2024-12-18T13:49:34+01:00",
          "tree_id": "0b3fc34b704dca4f183d131375f05b1033bb35b8",
          "url": "https://github.com/equinor/ert/commit/a6649a18d307b353de1e01428b189a6b335108e6"
        },
        "date": 1734526298593,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.2171199435754667,
            "unit": "iter/sec",
            "range": "stddev: 0.04954297387413186",
            "extra": "mean: 4.6057491703999975 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}