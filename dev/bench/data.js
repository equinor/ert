window.BENCHMARK_DATA = {
  "lastUpdate": 1709826468879,
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
          "id": "5ac8d3f5bf14fc423ed221d4a5c9a3aab1686067",
          "message": "Remove RowScaling\n\nThis was an experimental feature that was never used in production,\nonly in testing. We decided to go another path when it comes to\nimplementing distance based localization, so removing this.",
          "timestamp": "2024-03-01T08:37:40+01:00",
          "tree_id": "45b5102491e4fa9571f65463663883e35413794a",
          "url": "https://github.com/equinor/ert/commit/5ac8d3f5bf14fc423ed221d4a5c9a3aab1686067"
        },
        "date": 1709278825637,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.4115023815620079,
            "unit": "iter/sec",
            "range": "stddev: 0.46403284216903784",
            "extra": "mean: 2.4301195930000064 sec\nrounds: 5"
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
          "id": "f26a69942191141f1d29998f2fb4010655096502",
          "message": "Remove UpdateConfiguration\n\nThis was part of an experimental implementation of distance based\nlocalization, however it has problems in that it allowed the user\nto configure ert to update the same parameter multiple times.\nAdditionally it was not used in production, only in testing, and as\nsuch added a lot of unneeded complexity.",
          "timestamp": "2024-03-01T08:57:50+01:00",
          "tree_id": "b9b1159186fcb11511ef5e883ba756e75401b05f",
          "url": "https://github.com/equinor/ert/commit/f26a69942191141f1d29998f2fb4010655096502"
        },
        "date": 1709280052027,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19438542185214822,
            "unit": "iter/sec",
            "range": "stddev: 0.03634351675874752",
            "extra": "mean: 5.144418704200007 sec\nrounds: 5"
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
          "id": "a3c25e51f4bd2b164eeaee295f35848d58de2806",
          "message": "Remove unneeded jobname",
          "timestamp": "2024-03-01T09:59:58+01:00",
          "tree_id": "006c2d469c53b07de7e302f46ab3245ea0efc8d4",
          "url": "https://github.com/equinor/ert/commit/a3c25e51f4bd2b164eeaee295f35848d58de2806"
        },
        "date": 1709283777332,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18881468458567643,
            "unit": "iter/sec",
            "range": "stddev: 0.04688755847555915",
            "extra": "mean: 5.296198239000001 sec\nrounds: 5"
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
          "id": "0a4b5c5050dcf634e7ea4d1c28cfdbe90568768c",
          "message": "Pin pytest due to flaky\n\nhttps://github.com/box/flaky/issues/198",
          "timestamp": "2024-03-04T07:54:40+01:00",
          "tree_id": "babdde3945b36db64c73f929a9f8c1f99175f6a4",
          "url": "https://github.com/equinor/ert/commit/0a4b5c5050dcf634e7ea4d1c28cfdbe90568768c"
        },
        "date": 1709535484949,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18180068553175602,
            "unit": "iter/sec",
            "range": "stddev: 0.0718576727359095",
            "extra": "mean: 5.500529313600003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mtha@equinor.com",
            "name": "Matt Hall",
            "username": "kwinkunks"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a8053571810a1942ed852c9e47a8c93864236fda",
          "message": "Fix typos in ENKF_ALPHA docs (#6878)\n\n* Fix typos in ENKF_ALPHA docs ￼…\r\n\r\nUsing simpler and more consistent LaTeX with upright bold for vectors,\r\nwhich is a typical convention. Fixes equinor#6877.\r\n\r\n* Update docs/reference/configuration/keywords.rst\r\n\r\nCo-authored-by: Feda Curic <feda.curic@gmail.com>\r\n\r\n* Revisit the notation\r\n\r\nThe subscripts are awkward, but I think this is consistent.\r\n\r\n* Switch order of embellishments\r\n\r\nShould be the same result, but meaning is clearer\r\n\r\n---------\r\n\r\nCo-authored-by: Feda Curic <feda.curic@gmail.com>",
          "timestamp": "2024-03-04T11:43:03+01:00",
          "tree_id": "57138e4c7d27fbb093049f96ab7ef2a781338649",
          "url": "https://github.com/equinor/ert/commit/a8053571810a1942ed852c9e47a8c93864236fda"
        },
        "date": 1709549177872,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1914724128330193,
            "unit": "iter/sec",
            "range": "stddev: 0.05718233767800734",
            "extra": "mean: 5.222684486000015 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2a245c46190272a40e227a71724e0a6c69ac8dac",
          "message": "Mute transitions from PBS (#7334)\n\nAvoiding WARNING as logs from INFO and above are kept\r\ncentrally.",
          "timestamp": "2024-03-04T12:06:10+01:00",
          "tree_id": "b14f626f70caaaa97003141108e734fc475412fc",
          "url": "https://github.com/equinor/ert/commit/2a245c46190272a40e227a71724e0a6c69ac8dac"
        },
        "date": 1709550573613,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1907470851556616,
            "unit": "iter/sec",
            "range": "stddev: 0.04247751940240171",
            "extra": "mean: 5.242544069199994 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c385d319d312c2b0119937e134fc17b80a7dfd4a",
          "message": "Run LSF integration tests during Komodo tests (#7328)\n\nThis requires a real LSF cluster up and running and a \"bsub\"\r\ncommand in PATH.",
          "timestamp": "2024-03-04T12:07:24+01:00",
          "tree_id": "97907aa7fdc7d56ee24f9913a7f86cec6139728a",
          "url": "https://github.com/equinor/ert/commit/c385d319d312c2b0119937e134fc17b80a7dfd4a"
        },
        "date": 1709550632776,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19045834795361,
            "unit": "iter/sec",
            "range": "stddev: 0.04074040003150719",
            "extra": "mean: 5.250491830599993 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "ae1bdd5bbde2e96750bc776bef8cbe0782110377",
          "message": "Combine OpenPBS & LSF integration tests\n\nThere is a lot of overlap between the two drivers, so we combine some of\nthe generic integration tests into a single parameterised pytest.",
          "timestamp": "2024-03-04T14:20:52+01:00",
          "tree_id": "fd8d2b9d2be25986c3ce1d4b5941ac483b6e2cde",
          "url": "https://github.com/equinor/ert/commit/ae1bdd5bbde2e96750bc776bef8cbe0782110377"
        },
        "date": 1709558646023,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1869188932267656,
            "unit": "iter/sec",
            "range": "stddev: 0.06791015381687718",
            "extra": "mean: 5.349913979999997 sec\nrounds: 5"
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
          "id": "d45b78400ff8bc4e0bf8ec9fc61b5ea26a934a53",
          "message": "Upgrade setup python to v5 in github workflows",
          "timestamp": "2024-03-04T14:28:50+01:00",
          "tree_id": "2cd86dd7608571cdd99024f668e042d818a41b89",
          "url": "https://github.com/equinor/ert/commit/d45b78400ff8bc4e0bf8ec9fc61b5ea26a934a53"
        },
        "date": 1709559121263,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19136024153549663,
            "unit": "iter/sec",
            "range": "stddev: 0.04219778951530669",
            "extra": "mean: 5.22574591240001 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "7f656ef2854295a5ebc086e01eee4d721bae6a84",
          "message": "Add columns to Manage Cases overview",
          "timestamp": "2024-03-04T14:42:43+01:00",
          "tree_id": "8db022adcd086db844ad97bf6dd7f5bd62afa3bc",
          "url": "https://github.com/equinor/ert/commit/7f656ef2854295a5ebc086e01eee4d721bae6a84"
        },
        "date": 1709559948070,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18988369867153113,
            "unit": "iter/sec",
            "range": "stddev: 0.058724485371980537",
            "extra": "mean: 5.266381511399999 sec\nrounds: 5"
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
          "id": "126ac6b3c5bc8410625207b530fb94601cd0477e",
          "message": "Change from using flaky to pytest_rerunfailures (#7346)",
          "timestamp": "2024-03-04T16:26:23+01:00",
          "tree_id": "4ec69985206d95477e4f65a651c2323d26f6d412",
          "url": "https://github.com/equinor/ert/commit/126ac6b3c5bc8410625207b530fb94601cd0477e"
        },
        "date": 1709566184410,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19103852694284076,
            "unit": "iter/sec",
            "range": "stddev: 0.03664515502551673",
            "extra": "mean: 5.234546224799999 sec\nrounds: 5"
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
          "id": "2048dfee08da2a726f7503becfde3f77ab63318f",
          "message": "Build wheels for x86_64, intel and apple macOS\n\nFocus testing mostly on python 3.8, 3.11, 3.12\n\nSee table overview;\nhttps://github.com/equinor/ert/pull/7204#issuecomment-1953179760\n\nBrew hdf5 when macOS",
          "timestamp": "2024-03-05T10:48:12+01:00",
          "tree_id": "dd341046e722431ffb17d1f9abd037af2f440c20",
          "url": "https://github.com/equinor/ert/commit/2048dfee08da2a726f7503becfde3f77ab63318f"
        },
        "date": 1709632274104,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19088160117477426,
            "unit": "iter/sec",
            "range": "stddev: 0.028120685984427296",
            "extra": "mean: 5.238849600199989 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "04a246df352d51885da2a8c90939a93412c4c152",
          "message": "Remove leftover debug comment (#7362)",
          "timestamp": "2024-03-05T14:49:59Z",
          "tree_id": "d479ef0663995fc9911c1829f2ff6d9b09766a37",
          "url": "https://github.com/equinor/ert/commit/04a246df352d51885da2a8c90939a93412c4c152"
        },
        "date": 1709650404306,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19121782759672987,
            "unit": "iter/sec",
            "range": "stddev: 0.02733497664077433",
            "extra": "mean: 5.2296379086000115 sec\nrounds: 5"
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
          "id": "238073ae65fde0a2ce0e8fcb0bd4c0e496c5c858",
          "message": "Replace magic string <ERTCASE> in the runpath",
          "timestamp": "2024-03-06T09:43:49+02:00",
          "tree_id": "084db506792a802f877aa346ce5e4e5ee2f6f94d",
          "url": "https://github.com/equinor/ert/commit/238073ae65fde0a2ce0e8fcb0bd4c0e496c5c858"
        },
        "date": 1709711212007,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1925767130580511,
            "unit": "iter/sec",
            "range": "stddev: 0.031747132552301806",
            "extra": "mean: 5.192735840800003 sec\nrounds: 5"
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
          "id": "dff37c336d31fa6e60e2fb8f5619d2a812b3c022",
          "message": "Refactor load_parameters",
          "timestamp": "2024-03-06T08:51:19+01:00",
          "tree_id": "a4f31e7bf98c71feb9c8f8cc441b7de9844ccb67",
          "url": "https://github.com/equinor/ert/commit/dff37c336d31fa6e60e2fb8f5619d2a812b3c022"
        },
        "date": 1709711658724,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1915353187606801,
            "unit": "iter/sec",
            "range": "stddev: 0.04912953572589593",
            "extra": "mean: 5.2209692002 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "a0849763ae795000ac2e15ddeeffa1fa300ec1e6",
          "message": "Remove usages of Reader/Accessor alias\n\nAt one point there was a difference between Reader and Accessors. Now\nthey are the same class, with aliasing kept for compatibility. This is\nthe first step to remove the distinction in the Ert codebase.\n\nThe exact commands run were:\n```sh\nfind src -type f -name \"*.py\" -exec sed -i \"s,\\(Storage\\|Experiment\\|Ensemble\\)\\(Reader\\|Accessor\\),\\1,g\" '{}' \\;\nfind tests -type f -name \"*.py\" -exec sed -i \"s,\\(Storage\\|Experiment\\|Ensemble\\)\\(Reader\\|Accessor\\),\\1,g\" '{}' \\;\n```\n...and excepting `src/ert/storage/__init__.py`\n\nI also simplified a few instances of `Union[Ensemble, Ensemble]`",
          "timestamp": "2024-03-06T12:25:57+01:00",
          "tree_id": "09d2049a6bc02b4a2cd788133d817da032cc2208",
          "url": "https://github.com/equinor/ert/commit/a0849763ae795000ac2e15ddeeffa1fa300ec1e6"
        },
        "date": 1709724545085,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1936433714892122,
            "unit": "iter/sec",
            "range": "stddev: 0.040404550605539886",
            "extra": "mean: 5.164132354800017 sec\nrounds: 5"
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
          "id": "06da87b94743799e64d0616f00168e74dccd75cf",
          "message": "Migrate empty summary",
          "timestamp": "2024-03-06T12:48:58+01:00",
          "tree_id": "b0da48faa1e2ead56f6319eb0eb7f002080014f9",
          "url": "https://github.com/equinor/ert/commit/06da87b94743799e64d0616f00168e74dccd75cf"
        },
        "date": 1709725922602,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1872707213074148,
            "unit": "iter/sec",
            "range": "stddev: 0.02123018091554187",
            "extra": "mean: 5.339863022999987 sec\nrounds: 5"
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
          "id": "b20d565f42f3badf644b6fefce69a939cdb6e72e",
          "message": "Make parameter example test shrink better",
          "timestamp": "2024-03-06T16:12:45+01:00",
          "tree_id": "d9194d29aee837c0d89baa4212750fcabf7ed42b",
          "url": "https://github.com/equinor/ert/commit/b20d565f42f3badf644b6fefce69a939cdb6e72e"
        },
        "date": 1709738155036,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19232585619770642,
            "unit": "iter/sec",
            "range": "stddev: 0.034397690878340034",
            "extra": "mean: 5.199508894799999 sec\nrounds: 5"
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
          "id": "9b704c2b22ef28100105ab09981a365f8b515465",
          "message": "Add /global/bin to path for LSF tests in testkomodo.sh",
          "timestamp": "2024-03-06T16:26:25+01:00",
          "tree_id": "10eed88857e4c25ef50b22615ab03158e107ec6b",
          "url": "https://github.com/equinor/ert/commit/9b704c2b22ef28100105ab09981a365f8b515465"
        },
        "date": 1709738998996,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18588406694351728,
            "unit": "iter/sec",
            "range": "stddev: 0.05709947826334701",
            "extra": "mean: 5.379697229800013 sec\nrounds: 5"
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
          "id": "9d5d67bdc0fa79ec2743f78ec0d4d26a77f0bb40",
          "message": "Join 'create experiment' and 'case info' panels",
          "timestamp": "2024-03-07T08:26:35+01:00",
          "tree_id": "25b3dd4ceb0c273264530908bd4f14fc114ef9be",
          "url": "https://github.com/equinor/ert/commit/9d5d67bdc0fa79ec2743f78ec0d4d26a77f0bb40"
        },
        "date": 1709796581431,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19346906365730623,
            "unit": "iter/sec",
            "range": "stddev: 0.01928269667554579",
            "extra": "mean: 5.1687850299999925 sec\nrounds: 5"
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
          "id": "14a49199756e04706da3a467ce2cc39cf2ad4fa4",
          "message": "Use concurrency to cancel existing workflow jobs\n\nSetting concurrency groups for called workflows will cancel too much\nRemove concurrency from benchmark due to main only",
          "timestamp": "2024-03-07T09:00:10+01:00",
          "tree_id": "41e5b5217d1f40260cafc965070cea9e1b810ec2",
          "url": "https://github.com/equinor/ert/commit/14a49199756e04706da3a467ce2cc39cf2ad4fa4"
        },
        "date": 1709798588999,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18822459900637759,
            "unit": "iter/sec",
            "range": "stddev: 0.055211183852045265",
            "extra": "mean: 5.312801861600019 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "325361fef59e987effd8926b5a47f0904c6c39ca",
          "message": "Make ErtThread signal and re-raising opt-in\n\nThis commit does the following:\n1. Adds a function `set_signal_handler` that must be called before\n   re-raising works\n2. Adds a kwarg `should_raise` that must be True for re-raising to\n   work\n3. Workflow jobs don't reraise, only log",
          "timestamp": "2024-03-07T09:02:14+01:00",
          "tree_id": "6d672a633e5addc6b5e476c03a26993762994a9a",
          "url": "https://github.com/equinor/ert/commit/325361fef59e987effd8926b5a47f0904c6c39ca"
        },
        "date": 1709798716832,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1877525747872442,
            "unit": "iter/sec",
            "range": "stddev: 0.028153243413567078",
            "extra": "mean: 5.326158648600005 sec\nrounds: 5"
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
          "id": "53961ba4d2980da833e619694a6ba25720bb0e44",
          "message": "Fix bug where grdecl would output nan",
          "timestamp": "2024-03-07T09:16:51+01:00",
          "tree_id": "fbb6bd816d6b1f2ac922d4033b28d703527c3e3f",
          "url": "https://github.com/equinor/ert/commit/53961ba4d2980da833e619694a6ba25720bb0e44"
        },
        "date": 1709799599796,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19114311925654234,
            "unit": "iter/sec",
            "range": "stddev: 0.05175452951042588",
            "extra": "mean: 5.231681914000012 sec\nrounds: 5"
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
          "id": "748922cb4ab604b58f4e311049f0fa2f436343f5",
          "message": "Make qt error message for plot fail resizable+selectable",
          "timestamp": "2024-03-07T09:36:52+01:00",
          "tree_id": "0a669af1646320d0c67de29fbe79f20f399a7117",
          "url": "https://github.com/equinor/ert/commit/748922cb4ab604b58f4e311049f0fa2f436343f5"
        },
        "date": 1709800816022,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19025506089648997,
            "unit": "iter/sec",
            "range": "stddev: 0.04189905883627618",
            "extra": "mean: 5.256101967999996 sec\nrounds: 5"
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
          "id": "10550a6e93a41053c5856e7093bd54577acd8bc1",
          "message": "Autodetect num cpus/cores in pytest xdist",
          "timestamp": "2024-03-07T09:56:04+01:00",
          "tree_id": "d65209f7f7f5fd0a69562c16ebaa5b6c50e99cc3",
          "url": "https://github.com/equinor/ert/commit/10550a6e93a41053c5856e7093bd54577acd8bc1"
        },
        "date": 1709801939129,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1922751376347079,
            "unit": "iter/sec",
            "range": "stddev: 0.034169463093655365",
            "extra": "mean: 5.200880427400034 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ZOM@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "f8ede6f539dec5f04be18ce0ec3835ebb5c6edd9",
          "message": "OpenPBS: Treat 'E' state the same as 'F'\n\nBoth E and F mean that the process has finished. The difference seems\nthat jobs in E state might still be using the queue. This distinction is\nof no concern to us.",
          "timestamp": "2024-03-07T14:32:01+01:00",
          "tree_id": "79022cfd64b6bc27466bdff1178cea5a9cec5ed7",
          "url": "https://github.com/equinor/ert/commit/f8ede6f539dec5f04be18ce0ec3835ebb5c6edd9"
        },
        "date": 1709818507965,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1891673074510893,
            "unit": "iter/sec",
            "range": "stddev: 0.037157290806911734",
            "extra": "mean: 5.286325705400008 sec\nrounds: 5"
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
          "id": "43aa4dfa7bd61520e1033ca39cf9aec7155f907f",
          "message": "Do not retry when we get 'Job has finished' from qdel",
          "timestamp": "2024-03-07T15:05:28+01:00",
          "tree_id": "28af6ba416058e61b0d5d4d5394b83f2438e5487",
          "url": "https://github.com/equinor/ert/commit/43aa4dfa7bd61520e1033ca39cf9aec7155f907f"
        },
        "date": 1709820509411,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19260798518759042,
            "unit": "iter/sec",
            "range": "stddev: 0.17481567407492699",
            "extra": "mean: 5.191892740200001 sec\nrounds: 5"
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
          "id": "dd13514d59d1920742a9180cb88109740a8d5d6c",
          "message": "Add test of storage migration",
          "timestamp": "2024-03-07T15:31:31+01:00",
          "tree_id": "de88cb05d4ee5a792e66969e50fb3d4caf638b72",
          "url": "https://github.com/equinor/ert/commit/dd13514d59d1920742a9180cb88109740a8d5d6c"
        },
        "date": 1709822093350,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18760924165095055,
            "unit": "iter/sec",
            "range": "stddev: 0.08989905272022154",
            "extra": "mean: 5.33022782459999 sec\nrounds: 5"
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
          "id": "d33a48d88a22e58297cb1aeffe17d9aecb44e453",
          "message": "Fix calculation of batch size\n\n* Fix calculation of batch size\r\n\r\nNeed to take into account number of parameters.\r\n\r\n* Use float32 when calculating batch size",
          "timestamp": "2024-03-07T16:42:14+01:00",
          "tree_id": "3e25339bd43402d320c60ae6c9988e402ab91841",
          "url": "https://github.com/equinor/ert/commit/d33a48d88a22e58297cb1aeffe17d9aecb44e453"
        },
        "date": 1709826361504,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1437628396529341,
            "unit": "iter/sec",
            "range": "stddev: 0.13529948559944052",
            "extra": "mean: 6.955900442799793 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d260a0aa3ef5503e310acc8f94da799edd31535",
          "message": "Dodge bug when simulation_context tears down (#7354)\n\nThis dodges a bug in either simulation_context or in scheduler.py that\nis triggered by test_stop_sim(). This checks the behaviour when all\nrealizations are killed, and there is a bug that can cause this process\nto hang while trying to kill.\n\nThis commit will rerun the test in case of failure, and will make the CI\nwork while the underlying bug is being prioritized.",
          "timestamp": "2024-03-07T16:44:37+01:00",
          "tree_id": "7d281ebb35ebc777553698b6a55ac724eff92638",
          "url": "https://github.com/equinor/ert/commit/6d260a0aa3ef5503e310acc8f94da799edd31535"
        },
        "date": 1709826468327,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1412421837156037,
            "unit": "iter/sec",
            "range": "stddev: 0.03344512360659894",
            "extra": "mean: 7.0800378024 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}