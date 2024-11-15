window.BENCHMARK_DATA = {
  "lastUpdate": 1731674679890,
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
          "id": "b5d36717b7bdc15879f02c8a44ae614dea95a7eb",
          "message": "Add default values using Pandas assign in design_matrix",
          "timestamp": "2024-11-12T11:12:31+01:00",
          "tree_id": "5f970c12fb5113a4e6cf6080b1d79de5a2314a8f",
          "url": "https://github.com/equinor/ert/commit/b5d36717b7bdc15879f02c8a44ae614dea95a7eb"
        },
        "date": 1731406463776,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18967021305501305,
            "unit": "iter/sec",
            "range": "stddev: 0.054413618347240345",
            "extra": "mean: 5.272309151200005 sec\nrounds: 5"
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
          "id": "5b96e8a9a4cfcd809941596212ea637dcd493018",
          "message": "Add just command helper tool to repository",
          "timestamp": "2024-11-12T12:22:36+01:00",
          "tree_id": "ba4d5c9c3463b1b6dfa076ff25a6dd63bdc30150",
          "url": "https://github.com/equinor/ert/commit/5b96e8a9a4cfcd809941596212ea637dcd493018"
        },
        "date": 1731410677520,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.190592177302434,
            "unit": "iter/sec",
            "range": "stddev: 0.015545395168153927",
            "extra": "mean: 5.246805058600006 sec\nrounds: 5"
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
          "id": "f088b4b9140945a2effab4cfa38a42c4aac0c6c0",
          "message": "Remove macos fail flag export tests",
          "timestamp": "2024-11-12T14:36:03+01:00",
          "tree_id": "fdd56e044dfabfb8d111a9a721ad51c38566b218",
          "url": "https://github.com/equinor/ert/commit/f088b4b9140945a2effab4cfa38a42c4aac0c6c0"
        },
        "date": 1731418677331,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.195301811719276,
            "unit": "iter/sec",
            "range": "stddev: 0.008535186557729876",
            "extra": "mean: 5.120280202199996 sec\nrounds: 5"
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
          "id": "fecb69209de6b4751f75007404cf0ac04555e464",
          "message": "Add snapshot test for everest API",
          "timestamp": "2024-11-12T13:50:20Z",
          "tree_id": "229c71a030ca1bcc6d6b7299e6b372efc66ecbca",
          "url": "https://github.com/equinor/ert/commit/fecb69209de6b4751f75007404cf0ac04555e464"
        },
        "date": 1731419533395,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19151137827394382,
            "unit": "iter/sec",
            "range": "stddev: 0.02492280900488543",
            "extra": "mean: 5.221621864000002 sec\nrounds: 5"
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
          "id": "e50f2d3463133db536e3204cd7c27b528122a4e7",
          "message": "Assume non-LSF host error is flaky\n\nThe LSF driver experiences crashes stemming from bsub returning with the\nerror message 'Request from non-LSF host rejected'. There are reasons to\nbelieve this is not a permanent error, but some flakyness in the IP\ninfrastructure, and thus should should be categorized as a retriable\nfailure.\n\nThe reason for believing this is flakyness is mostly from the fact that\nthe same error is also seen on 'bjobs'-calls. If it was a permanent\nfailure scenario, there would be an enourmous amount of error from these\nbjobs calls, but there is not.",
          "timestamp": "2024-11-12T14:55:36+01:00",
          "tree_id": "bf29e38b8c6c51bab62bc55819496f2308075b34",
          "url": "https://github.com/equinor/ert/commit/e50f2d3463133db536e3204cd7c27b528122a4e7"
        },
        "date": 1731419847936,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1855084750782697,
            "unit": "iter/sec",
            "range": "stddev: 0.12035984176026342",
            "extra": "mean: 5.3905892956 sec\nrounds: 5"
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
          "id": "efb812cf823994775f4ddea84a3c19b889acf9dd",
          "message": "Add trace ID to clipboard debug info and title bar (#9157)",
          "timestamp": "2024-11-12T15:05:33+01:00",
          "tree_id": "be794f114365a4a6196b469bd1ad04134adc2c69",
          "url": "https://github.com/equinor/ert/commit/efb812cf823994775f4ddea84a3c19b889acf9dd"
        },
        "date": 1731420442192,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19307666222800574,
            "unit": "iter/sec",
            "range": "stddev: 0.01839081835385295",
            "extra": "mean: 5.179289865800001 sec\nrounds: 5"
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
          "id": "22b7f7142ca8232fb9c5f39fbfb99cfc58aca5fa",
          "message": "Fixup ert plugin documentation\n\n- Specifies package structure assumption\n- Adds pyproject.toml example\n- Fixes syntax error in code example",
          "timestamp": "2024-11-12T15:51:10+01:00",
          "tree_id": "cf7a53afb52243068701175efdb21ae6700ff4bc",
          "url": "https://github.com/equinor/ert/commit/22b7f7142ca8232fb9c5f39fbfb99cfc58aca5fa"
        },
        "date": 1731423238294,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1927639173767591,
            "unit": "iter/sec",
            "range": "stddev: 0.0368780184250326",
            "extra": "mean: 5.187692871200005 sec\nrounds: 5"
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
          "id": "93081a13cfba2dd439943a5a72912d786e3e2890",
          "message": "Move everserver config to ServerConfig",
          "timestamp": "2024-11-13T07:48:44+01:00",
          "tree_id": "a1cc3c08e11020bf6233519842e48d7e4c985a60",
          "url": "https://github.com/equinor/ert/commit/93081a13cfba2dd439943a5a72912d786e3e2890"
        },
        "date": 1731480640195,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19134390136318888,
            "unit": "iter/sec",
            "range": "stddev: 0.01955277528329969",
            "extra": "mean: 5.226192174799996 sec\nrounds: 5"
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
          "id": "03cfa25311451e196ded467b2e260b41a79d6587",
          "message": "Remove old job queue test snapshot",
          "timestamp": "2024-11-13T07:59:48+01:00",
          "tree_id": "932ccdc696aefc5c0347d0d0f614718ebec01a35",
          "url": "https://github.com/equinor/ert/commit/03cfa25311451e196ded467b2e260b41a79d6587"
        },
        "date": 1731481306024,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19111658429816888,
            "unit": "iter/sec",
            "range": "stddev: 0.030203551263009",
            "extra": "mean: 5.232408289800003 sec\nrounds: 5"
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
          "id": "5ac831ed67f2d15393bdc9fa2da53a5014b5e3e2",
          "message": "Add warning when everest-models file outputs do not match everest objectives",
          "timestamp": "2024-11-13T18:27:54+09:00",
          "tree_id": "96be8224578478c256319794909450a7966210c4",
          "url": "https://github.com/equinor/ert/commit/5ac831ed67f2d15393bdc9fa2da53a5014b5e3e2"
        },
        "date": 1731490188870,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19508941088438342,
            "unit": "iter/sec",
            "range": "stddev: 0.02441490217600472",
            "extra": "mean: 5.1258548347999975 sec\nrounds: 5"
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
          "id": "d673a07d7e4d8db7f1e49ded19bc1739e3dec249",
          "message": "Make ropt use \"_\" instead of \".\" delimiter",
          "timestamp": "2024-11-13T11:26:15+01:00",
          "tree_id": "1147ceee05d07a5dd6382b7a21e8182fc6807d53",
          "url": "https://github.com/equinor/ert/commit/d673a07d7e4d8db7f1e49ded19bc1739e3dec249"
        },
        "date": 1731493691047,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19402654589616652,
            "unit": "iter/sec",
            "range": "stddev: 0.04883771372788808",
            "extra": "mean: 5.153933939200004 sec\nrounds: 5"
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
          "id": "d11ba38bd8f93808ae60d320e03ab10725294dec",
          "message": "Use uvloop for asyncio\n\nThis replacement loop is significantly faster for sockets and streams",
          "timestamp": "2024-11-13T12:10:35+01:00",
          "tree_id": "edd7b884d97ea1c00fce1f6c50255b0a31c14538",
          "url": "https://github.com/equinor/ert/commit/d11ba38bd8f93808ae60d320e03ab10725294dec"
        },
        "date": 1731496351881,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1952211746880142,
            "unit": "iter/sec",
            "range": "stddev: 0.023524822419013096",
            "extra": "mean: 5.122395158199998 sec\nrounds: 5"
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
          "id": "648aa27f158115786ed1d80a6cbd17f669037150",
          "message": "Fix error with inconsistent blobsize in flaky test\n\nFixing this bug will probably mitigate flakyness",
          "timestamp": "2024-11-13T12:18:26+01:00",
          "tree_id": "f38f6371c61cf5a742d86df9bf9eddbbb5a06537",
          "url": "https://github.com/equinor/ert/commit/648aa27f158115786ed1d80a6cbd17f669037150"
        },
        "date": 1731496818921,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18871684404487493,
            "unit": "iter/sec",
            "range": "stddev: 0.01539583628591243",
            "extra": "mean: 5.298944061200018 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "114403625+andreas-el@users.noreply.github.com",
            "name": "Andreas Eknes Lie",
            "username": "andreas-el"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f6543ff9f57948ce6d65f1e3c8f40da98f8ed9f",
          "message": "Add right click capability to open external plot windows",
          "timestamp": "2024-11-13T12:30:18+01:00",
          "tree_id": "58dee35ba649edec11f47f8b118b28f4402c2303",
          "url": "https://github.com/equinor/ert/commit/6f6543ff9f57948ce6d65f1e3c8f40da98f8ed9f"
        },
        "date": 1731497528855,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1904725305954941,
            "unit": "iter/sec",
            "range": "stddev: 0.028072447011939037",
            "extra": "mean: 5.25010087739999 sec\nrounds: 5"
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
          "id": "8903abb42591d50c9c26b2ad086609b0286baf8d",
          "message": "Add egg snapshots for py38 and py311",
          "timestamp": "2024-11-13T12:32:25+01:00",
          "tree_id": "16931008c0b9b6cdcfefcd7e9a23df0e89140785",
          "url": "https://github.com/equinor/ert/commit/8903abb42591d50c9c26b2ad086609b0286baf8d"
        },
        "date": 1731497655119,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19058716203305348,
            "unit": "iter/sec",
            "range": "stddev: 0.017363024075441224",
            "extra": "mean: 5.246943127399999 sec\nrounds: 5"
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
          "id": "c5c693ead24db25a09d27178b44c59bbbd0bf5b1",
          "message": "Add return type annotation",
          "timestamp": "2024-11-13T13:02:20+01:00",
          "tree_id": "316da7e035920d8206bcd2c75a9a739a673b5952",
          "url": "https://github.com/equinor/ert/commit/c5c693ead24db25a09d27178b44c59bbbd0bf5b1"
        },
        "date": 1731499456831,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19187204614061862,
            "unit": "iter/sec",
            "range": "stddev: 0.041618329787992694",
            "extra": "mean: 5.211806618599996 sec\nrounds: 5"
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
          "id": "d32d419d487dcd9b0fdcd3dc3369d741dd1a41e7",
          "message": "Log when dark storage fails to load response",
          "timestamp": "2024-11-13T13:32:12+01:00",
          "tree_id": "0ff1ac4db4aa0ac92aa5e9234439d23af1844b55",
          "url": "https://github.com/equinor/ert/commit/d32d419d487dcd9b0fdcd3dc3369d741dd1a41e7"
        },
        "date": 1731501245742,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19224891034741187,
            "unit": "iter/sec",
            "range": "stddev: 0.01960705610632294",
            "extra": "mean: 5.201589950200008 sec\nrounds: 5"
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
          "id": "a1a028c3aacbd0ba1e813b4c9842bbaf108ae66d",
          "message": "Disregard expected_objectives from OptimalResults\n\nthe calculation within seba does not make sense",
          "timestamp": "2024-11-14T09:03:31+01:00",
          "tree_id": "434623a1d3e6a0ff07e83e6005c3dbfa8a35da2b",
          "url": "https://github.com/equinor/ert/commit/a1a028c3aacbd0ba1e813b4c9842bbaf108ae66d"
        },
        "date": 1731571528260,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19223385464088083,
            "unit": "iter/sec",
            "range": "stddev: 0.02040400479608584",
            "extra": "mean: 5.201997337399996 sec\nrounds: 5"
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
          "id": "1b235cf54a592a6b3be1b0c7ba0a95570115d288",
          "message": "Use sh for all code blocks in readme",
          "timestamp": "2024-11-14T10:26:21+01:00",
          "tree_id": "5a88f390f18b00da23698eb80ede1d040f3809c9",
          "url": "https://github.com/equinor/ert/commit/1b235cf54a592a6b3be1b0c7ba0a95570115d288"
        },
        "date": 1731576497317,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18999392828727604,
            "unit": "iter/sec",
            "range": "stddev: 0.01782326892716323",
            "extra": "mean: 5.263326091600004 sec\nrounds: 5"
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
          "id": "fd6fa0574a8edc085d50d2872d16b19646217dfd",
          "message": "Fix typing for batching interval",
          "timestamp": "2024-11-14T10:42:41+01:00",
          "tree_id": "6f55cfa43074a52a9439ac804c8563b1253dd420",
          "url": "https://github.com/equinor/ert/commit/fd6fa0574a8edc085d50d2872d16b19646217dfd"
        },
        "date": 1731577474307,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19463709917666028,
            "unit": "iter/sec",
            "range": "stddev: 0.010724499538523202",
            "extra": "mean: 5.137766665400005 sec\nrounds: 5"
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
          "id": "91310f99e95e135cc2e63fbce0640033352e9c7b",
          "message": "Run ruff",
          "timestamp": "2024-11-14T11:54:05+01:00",
          "tree_id": "b9864e9c010671e7d6344582cb8b114e47bf6034",
          "url": "https://github.com/equinor/ert/commit/91310f99e95e135cc2e63fbce0640033352e9c7b"
        },
        "date": 1731581777665,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18832109962684743,
            "unit": "iter/sec",
            "range": "stddev: 0.06783166517792047",
            "extra": "mean: 5.310079444000007 sec\nrounds: 5"
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
          "id": "9a3c864607c891791a023e54abe6cea4706bf9ae",
          "message": "Convert ErtConfig to dataclass",
          "timestamp": "2024-11-14T12:28:20+01:00",
          "tree_id": "9afa168d7d375b58f8ba9a20880ff18dafa4596c",
          "url": "https://github.com/equinor/ert/commit/9a3c864607c891791a023e54abe6cea4706bf9ae"
        },
        "date": 1731583824251,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1924551332956283,
            "unit": "iter/sec",
            "range": "stddev: 0.03686378717714496",
            "extra": "mean: 5.196016249999997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sted@equinor.com",
            "name": "StephanDeHoop",
            "username": "StephanDeHoop"
          },
          "committer": {
            "email": "stephan.dehoop@tno.nl",
            "name": "Stephan de Hoop",
            "username": "StephanDeHoop"
          },
          "distinct": true,
          "id": "137bdc64b0be0f9d300720e487458226dc4ee1e4",
          "message": "Use everest.strings instead of hardcoded",
          "timestamp": "2024-11-14T12:52:17+01:00",
          "tree_id": "6ef924718b0ff0a5b4da73391879d4072fb7d11d",
          "url": "https://github.com/equinor/ert/commit/137bdc64b0be0f9d300720e487458226dc4ee1e4"
        },
        "date": 1731585249027,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1954745562795878,
            "unit": "iter/sec",
            "range": "stddev: 0.021056245468429068",
            "extra": "mean: 5.115755313800008 sec\nrounds: 5"
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
            "email": "dan.sava42@gmail.com",
            "name": "Dan Sava",
            "username": "DanSava"
          },
          "distinct": true,
          "id": "120659c388e9419327c899d34f1c4a346b551037",
          "message": "Adds test for invalid install_data templates",
          "timestamp": "2024-11-14T21:11:05+09:00",
          "tree_id": "0c0948dcaf4f5dee5b224c55944eec961484d2cf",
          "url": "https://github.com/equinor/ert/commit/120659c388e9419327c899d34f1c4a346b551037"
        },
        "date": 1731586385230,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1919442611014824,
            "unit": "iter/sec",
            "range": "stddev: 0.029696380098208634",
            "extra": "mean: 5.209845786800014 sec\nrounds: 5"
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
          "id": "79aad486f76bbb27746c34de4c29758b028a93a4",
          "message": "Fix bug where Everest would not start with relative paths",
          "timestamp": "2024-11-14T13:22:04+01:00",
          "tree_id": "15e370783b3ae0b9b823688d0576f5f9acef1810",
          "url": "https://github.com/equinor/ert/commit/79aad486f76bbb27746c34de4c29758b028a93a4"
        },
        "date": 1731587032787,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19476826957680538,
            "unit": "iter/sec",
            "range": "stddev: 0.027146648481339533",
            "extra": "mean: 5.134306538600003 sec\nrounds: 5"
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
          "id": "9819a5b03255355abd69d16667bd95ae69e0afbc",
          "message": "Fix flaky rightclick plot-button test",
          "timestamp": "2024-11-14T13:40:12+01:00",
          "tree_id": "77d974138c33d242d929ff7cd5a11ae2e83b941e",
          "url": "https://github.com/equinor/ert/commit/9819a5b03255355abd69d16667bd95ae69e0afbc"
        },
        "date": 1731588134861,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19231872867327135,
            "unit": "iter/sec",
            "range": "stddev: 0.02460795550748442",
            "extra": "mean: 5.199701593800006 sec\nrounds: 5"
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
          "id": "e544ddcdc19898ef7aa6a8a198ebd4fdcf99e086",
          "message": "Fix faulty validation for maintained forward model objectives.",
          "timestamp": "2024-11-15T15:45:23+09:00",
          "tree_id": "7d5cff4b0ff8f6749f7ba2f42269159f2831d373",
          "url": "https://github.com/equinor/ert/commit/e544ddcdc19898ef7aa6a8a198ebd4fdcf99e086"
        },
        "date": 1731653254257,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18910558466570274,
            "unit": "iter/sec",
            "range": "stddev: 0.08942009358555235",
            "extra": "mean: 5.288051126400001 sec\nrounds: 5"
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
          "id": "ffaa27ab2532153f0971dab394bf825c1b56ad54",
          "message": "Extend save parameters to handle multiple realizations\n\nAdd test that uses the new functionality and also documents\r\nsome troublesome behavior of adaptive localization.",
          "timestamp": "2024-11-15T09:45:29+01:00",
          "tree_id": "a487246ff4b0bf535714a7434410a52ae809d1b3",
          "url": "https://github.com/equinor/ert/commit/ffaa27ab2532153f0971dab394bf825c1b56ad54"
        },
        "date": 1731660444481,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19541078065371262,
            "unit": "iter/sec",
            "range": "stddev: 0.008124962162825272",
            "extra": "mean: 5.117424927399986 sec\nrounds: 5"
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
          "id": "76125352e552ab45a2ff9c6c0a6f6c8745b7adf5",
          "message": "Fetch all tags in readthedocs workflow",
          "timestamp": "2024-11-15T11:48:13+01:00",
          "tree_id": "e33af86cc33b24b14b1ec322a10661ec8c7bead1",
          "url": "https://github.com/equinor/ert/commit/76125352e552ab45a2ff9c6c0a6f6c8745b7adf5"
        },
        "date": 1731667816815,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19012768757649334,
            "unit": "iter/sec",
            "range": "stddev: 0.041726213055851856",
            "extra": "mean: 5.2596232182 sec\nrounds: 5"
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
          "id": "a1a13cc1eb0aa3b8ac77bdc995f414c5e3428c08",
          "message": "Simplify test logic",
          "timestamp": "2024-11-15T21:42:46+09:00",
          "tree_id": "419e2444034fb044dc4d8dc1dca0706fee4905aa",
          "url": "https://github.com/equinor/ert/commit/a1a13cc1eb0aa3b8ac77bdc995f414c5e3428c08"
        },
        "date": 1731674679407,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/ert/performance_tests/test_analysis.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19076791500288928,
            "unit": "iter/sec",
            "range": "stddev: 0.02509802690880647",
            "extra": "mean: 5.241971638599995 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}