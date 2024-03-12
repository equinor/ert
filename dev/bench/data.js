window.BENCHMARK_DATA = {
  "lastUpdate": 1710229354651,
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
      },
      {
        "commit": {
          "author": {
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "committer": {
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "da3f122acfae4a7274a787c71e844c4e3d74d974",
          "message": "Upgrade submodule",
          "timestamp": "2024-03-07T17:24:33+01:00",
          "tree_id": "8e6730bfb2495bdb5f595e60e0972e9181d961c6",
          "url": "https://github.com/equinor/ert/commit/da3f122acfae4a7274a787c71e844c4e3d74d974"
        },
        "date": 1709828886127,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.13564243031652687,
            "unit": "iter/sec",
            "range": "stddev: 0.05020327753452046",
            "extra": "mean: 7.372324409600014 sec\nrounds: 5"
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
          "id": "b1fc33adcfdb5df473ab6eab302e562846b2d578",
          "message": "Remove -s from pytest in CI\n\n`pytest-xdist` is incompatible with `-s`, and the output from successful\ntests isn't interesting.\n\nRef: https://github.com/pytest-dev/pytest-xdist/blob/017cc72b7090dc4bb7e7ad3d0caab024feb977a8/docs/known-limitations.rst#output-stdout-and-stderr-from-workers",
          "timestamp": "2024-03-07T20:22:38+01:00",
          "tree_id": "94447fb9b6b1a85e74411f9ef220d478b913f8f7",
          "url": "https://github.com/equinor/ert/commit/b1fc33adcfdb5df473ab6eab302e562846b2d578"
        },
        "date": 1709839561567,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.14140946083526335,
            "unit": "iter/sec",
            "range": "stddev: 0.039355385383929506",
            "extra": "mean: 7.071662632000004 sec\nrounds: 5"
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
          "id": "5b3f0702f87d32955a58cc0b7991a55afdce582a",
          "message": "Avoid \"qstat -f\" for non-finished jobs in PBS (#7376)\n\nqstat -f (f for \"full format\") is a heavy operation for the PBS cluster,\r\nand we only use it to obtain the Exit status for the job.\r\n\r\nThis will change the polling into using qstat -f only for jobs\r\nthat are already marked as finished, through polling with\r\nthe default qstat output.",
          "timestamp": "2024-03-08T06:42:37Z",
          "tree_id": "d71af2ef8783b09356126838b9ec28513c77ec32",
          "url": "https://github.com/equinor/ert/commit/5b3f0702f87d32955a58cc0b7991a55afdce582a"
        },
        "date": 1709880353352,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1405057775419177,
            "unit": "iter/sec",
            "range": "stddev: 0.03569247558360399",
            "extra": "mean: 7.117145056200025 sec\nrounds: 5"
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
          "id": "d015dfa8997a21dd385e5267b4b0908f2fcd3631",
          "message": "Revert calculation of batch size fix\n\nReverts this commit but adds derivation of formula to not make the same mistake again:\r\nd33a48d88a22e58297cb1aeffe17d9aecb44e453",
          "timestamp": "2024-03-08T09:19:24+01:00",
          "tree_id": "8b30f9169faec354b136bd944d651f8dc02b1478",
          "url": "https://github.com/equinor/ert/commit/d015dfa8997a21dd385e5267b4b0908f2fcd3631"
        },
        "date": 1709886175993,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1904727588899406,
            "unit": "iter/sec",
            "range": "stddev: 0.05122275832953483",
            "extra": "mean: 5.250094584800036 sec\nrounds: 5"
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
          "id": "bd37ca9a32629a1fa67f00dd47a1817e7b20081c",
          "message": "Revert \"Rename Field to FieldConfig\"\n\nThis reverts commit 119c8731cbd2677f10ba1aa0be8502603eb9b6c5.",
          "timestamp": "2024-03-08T09:45:08+01:00",
          "tree_id": "b4f0e6eca6d806ec731adda588a8aff42ed9a276",
          "url": "https://github.com/equinor/ert/commit/bd37ca9a32629a1fa67f00dd47a1817e7b20081c"
        },
        "date": 1709887693376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1912317401900545,
            "unit": "iter/sec",
            "range": "stddev: 0.05409308758257642",
            "extra": "mean: 5.229257439199978 sec\nrounds: 5"
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
          "id": "006499cff7d98620446c45872d8d92985dd512bb",
          "message": "Ensure no duplicate keys in SummaryConfig",
          "timestamp": "2024-03-08T12:25:52+01:00",
          "tree_id": "0365d3aa47e52a53eb16f5debb4a01ec8c1f12b8",
          "url": "https://github.com/equinor/ert/commit/006499cff7d98620446c45872d8d92985dd512bb"
        },
        "date": 1709897329769,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19094525803545884,
            "unit": "iter/sec",
            "range": "stddev: 0.02355417950002718",
            "extra": "mean: 5.237103085399997 sec\nrounds: 5"
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
          "id": "2f020324c51de074153dba7b6f7704073d1c366b",
          "message": "Add more cases to storage test",
          "timestamp": "2024-03-08T13:51:28+01:00",
          "tree_id": "1d2838ec14ea8c3cd619b4946762e572d46b6320",
          "url": "https://github.com/equinor/ert/commit/2f020324c51de074153dba7b6f7704073d1c366b"
        },
        "date": 1709902475605,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19223083853674058,
            "unit": "iter/sec",
            "range": "stddev: 0.03558162363451653",
            "extra": "mean: 5.202078956800017 sec\nrounds: 5"
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
          "id": "6d9656b344fe6feaa06b26b5d68f82a1e13c90dc",
          "message": "Treat exception in long lived tasks by cancelling the jobs correctly\n\nWe gather collectively results of realization tasks and scheduling_tasks\n(long live scheduling tasks). The main point is that the exceptions from realization tasks are treated differently than\nthe exceptions from scheduling tasks. Exception in scheduling tasks requires immidiate handling.\n\nThis includes unit tests when OpenPBS driver hanging / fails and\nscheduler related exception tests.\n\nAdditionally, this commit removes async_utils.background_tasks and test_async_utils.py\n\nCo-authored-by: Jonathan Karlsen <jonak@equinor.com>",
          "timestamp": "2024-03-10T16:45:41+01:00",
          "tree_id": "baea8d285c9011453884299ac96248b95e3a0f55",
          "url": "https://github.com/equinor/ert/commit/6d9656b344fe6feaa06b26b5d68f82a1e13c90dc"
        },
        "date": 1710085721837,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18745106016474103,
            "unit": "iter/sec",
            "range": "stddev: 0.08233757578141407",
            "extra": "mean: 5.334725763200015 sec\nrounds: 5"
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
          "id": "bbcb52bff3f2903a2bf55005e4992c2b0f628171",
          "message": "Fix external_ert_script does not fail on error (#7213)\n\n* Improve readability cli test test_that_stop_on_fail_workflow_jobs_stop_ert\r\n\r\n* Fix external_ert_script does not fail on error\r\n\r\n* Remove STOP_ON_FAIL keyword from scripts",
          "timestamp": "2024-03-11T10:15:23+01:00",
          "tree_id": "5b5fde9be6bcc61c02e3988d0a82c41b9a4a2ada",
          "url": "https://github.com/equinor/ert/commit/bbcb52bff3f2903a2bf55005e4992c2b0f628171"
        },
        "date": 1710148719294,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18888618221999529,
            "unit": "iter/sec",
            "range": "stddev: 0.08422267050500042",
            "extra": "mean: 5.294193509800005 sec\nrounds: 5"
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
          "id": "d09bd4e400db30d8ad612877b01016dfdd7f7c4b",
          "message": "Use newer style v resource allocation for Torque (C) and PBS (Python) (#7389)\n\nUse newer style resource allocation for qsub\r\n\r\nIn short; nodes replaced by select, and ppn replaced by ncpus.",
          "timestamp": "2024-03-11T12:45:22+01:00",
          "tree_id": "bd032e03c74c724b09c09b2f6e79ced0447f08fa",
          "url": "https://github.com/equinor/ert/commit/d09bd4e400db30d8ad612877b01016dfdd7f7c4b"
        },
        "date": 1710157712083,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1911255445449341,
            "unit": "iter/sec",
            "range": "stddev: 0.026319712460644306",
            "extra": "mean: 5.232162986800006 sec\nrounds: 5"
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
          "id": "6bac0ba8d54f6133c8017ca0ed55f7b2670c2084",
          "message": "Use a compiled regex for matching in summary\n\nlooping over a fnmatch was simply too slow",
          "timestamp": "2024-03-11T15:03:45+01:00",
          "tree_id": "0615b32af143c921ed2e7e2b5af6955462df06b8",
          "url": "https://github.com/equinor/ert/commit/6bac0ba8d54f6133c8017ca0ed55f7b2670c2084"
        },
        "date": 1710166011955,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19082239886385385,
            "unit": "iter/sec",
            "range": "stddev: 0.020303577191101322",
            "extra": "mean: 5.240474943999999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "sonso@equinor.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "committer": {
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "802375058dbcc08709945c84088133b7375ac935",
          "message": "Use block storage path fixture in storage test",
          "timestamp": "2024-03-12T08:39:36+01:00",
          "tree_id": "6ece5e6fdc73f8ad0b7aafa60807300f217d57ed",
          "url": "https://github.com/equinor/ert/commit/802375058dbcc08709945c84088133b7375ac935"
        },
        "date": 1710229354182,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1918157586620159,
            "unit": "iter/sec",
            "range": "stddev: 0.024688740907438094",
            "extra": "mean: 5.2133360000000035 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}