window.BENCHMARK_DATA = {
  "lastUpdate": 1710836910446,
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
          "id": "d22053cf3ebd5c68a890dbdc4256df4881d25375",
          "message": "Show total time in update tab",
          "timestamp": "2024-03-13T09:59:56+01:00",
          "tree_id": "072e047752c118db36e7e6ac29b1413b56273dbc",
          "url": "https://github.com/equinor/ert/commit/d22053cf3ebd5c68a890dbdc4256df4881d25375"
        },
        "date": 1710320574011,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19222737731400147,
            "unit": "iter/sec",
            "range": "stddev: 0.016927445080984736",
            "extra": "mean: 5.202172624800005 sec\nrounds: 5"
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
          "id": "8717b4f24a9eb5d1a5bd3af8a92123ac62360792",
          "message": "Rename simulation arguments",
          "timestamp": "2024-03-13T10:00:38+01:00",
          "tree_id": "2e1e62cf25303f483e5871409ffce002f214bb0f",
          "url": "https://github.com/equinor/ert/commit/8717b4f24a9eb5d1a5bd3af8a92123ac62360792"
        },
        "date": 1710320641892,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1857615228521758,
            "unit": "iter/sec",
            "range": "stddev: 0.09841114640663373",
            "extra": "mean: 5.383246135400031 sec\nrounds: 5"
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
          "id": "5e5ed6f8c20ff4294eaf1e9f731a88046eb922f6",
          "message": "Add more info about observations",
          "timestamp": "2024-03-13T10:25:50+01:00",
          "tree_id": "4c58745ddcf15d36b06dee39c514af308dbc5624",
          "url": "https://github.com/equinor/ert/commit/5e5ed6f8c20ff4294eaf1e9f731a88046eb922f6"
        },
        "date": 1710322162848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18997134500651605,
            "unit": "iter/sec",
            "range": "stddev: 0.03260841325770118",
            "extra": "mean: 5.263951781600008 sec\nrounds: 5"
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
          "id": "e65d1680b17055c81626e6a5a7c870d6aae59045",
          "message": "Parameterize test",
          "timestamp": "2024-03-13T12:20:06+01:00",
          "tree_id": "6fc07e0b15a750cb0d6934ca30660212951ea7eb",
          "url": "https://github.com/equinor/ert/commit/e65d1680b17055c81626e6a5a7c870d6aae59045"
        },
        "date": 1710328986534,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1870301541853596,
            "unit": "iter/sec",
            "range": "stddev: 0.13456434456725036",
            "extra": "mean: 5.346731409999973 sec\nrounds: 5"
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
          "id": "3381e679a4a9b5c436dd3c4701ffbd565301db0b",
          "message": "Deprecate unused lsf options (#7428)\n\n* Add deprecation warnings for outdated LSF queue options\r\n\r\nThis commit adds deprecation warnings for the outdated LSF queue options: LSF_LOGIN_SHELL, LSF_RSH_CMD.\r\n\r\n* Fix config schema not handling duplicate keyword deprecations\r\n\r\nThis commit makes it so that one keyword (for example QUEUE_CONFIG) can\r\nhave multiple deprecation warnings.\r\n\r\n* Remove deprecated LSF queue options from docs\r\n\r\nThis commit removed the deprecated LSF queue options: LSF_LOGIN_SHELL and LSF_RSH_CMD from the docs.",
          "timestamp": "2024-03-13T13:30:32+01:00",
          "tree_id": "559f4784f614a0109c44b1ec233792fc07a5b82d",
          "url": "https://github.com/equinor/ert/commit/3381e679a4a9b5c436dd3c4701ffbd565301db0b"
        },
        "date": 1710333212792,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1896308087239947,
            "unit": "iter/sec",
            "range": "stddev: 0.042357940057563125",
            "extra": "mean: 5.273404710599993 sec\nrounds: 5"
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
          "id": "2a451b8da610fe32df0f62f213fe2dcf8cca43dd",
          "message": "Add migration for ert_kind",
          "timestamp": "2024-03-13T14:34:48+01:00",
          "tree_id": "443d49677be21509828d4829305dfecc6d0163d6",
          "url": "https://github.com/equinor/ert/commit/2a451b8da610fe32df0f62f213fe2dcf8cca43dd"
        },
        "date": 1710337072444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1885534016903004,
            "unit": "iter/sec",
            "range": "stddev: 0.02173812389661609",
            "extra": "mean: 5.303537305799995 sec\nrounds: 5"
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
          "id": "eee4d2ac2e2b0a26c08d84208e6f794d5b89d51b",
          "message": "Include LocalDriver in generic driver tests",
          "timestamp": "2024-03-13T15:25:09+01:00",
          "tree_id": "967054f6b3d650c2708cf1974ecb6e3989edc43b",
          "url": "https://github.com/equinor/ert/commit/eee4d2ac2e2b0a26c08d84208e6f794d5b89d51b"
        },
        "date": 1710340088860,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19127202184314754,
            "unit": "iter/sec",
            "range": "stddev: 0.08026117134082517",
            "extra": "mean: 5.228156164000029 sec\nrounds: 5"
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
            "email": "ejah@equinor.com",
            "name": "Eivind Jahren",
            "username": "eivindjahren"
          },
          "distinct": true,
          "id": "a0a93416c20b7e4739c5f4a4ed015c04b4a4a5a6",
          "message": "Only build & test using macos runners for tags",
          "timestamp": "2024-03-13T16:57:07+01:00",
          "tree_id": "a75b10aa57d0fb1c2a82e9d53fe2042b2d3efda4",
          "url": "https://github.com/equinor/ert/commit/a0a93416c20b7e4739c5f4a4ed015c04b4a4a5a6"
        },
        "date": 1710345620046,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18547923309362258,
            "unit": "iter/sec",
            "range": "stddev: 0.03759214094862649",
            "extra": "mean: 5.391439156400002 sec\nrounds: 5"
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
          "id": "9bdea6785b011fb8a1621e050d50b33886a90f85",
          "message": "Account for cases when the summary data is empty.\n\nCatch also KeyError when checking summary_data[\"name\"].values",
          "timestamp": "2024-03-14T09:29:01+02:00",
          "tree_id": "b137a35007a14ce624c98d3fc3b268fb0bb947c0",
          "url": "https://github.com/equinor/ert/commit/9bdea6785b011fb8a1621e050d50b33886a90f85"
        },
        "date": 1710401537686,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19221563091958727,
            "unit": "iter/sec",
            "range": "stddev: 0.038374593394346984",
            "extra": "mean: 5.202490532199988 sec\nrounds: 5"
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
            "email": "44577479+oyvindeide@users.noreply.github.com",
            "name": "Øyvind Eide",
            "username": "oyvindeide"
          },
          "distinct": true,
          "id": "d72d2fd9451caceebcb3ea7d48102af60266efc4",
          "message": "Type using StringList to make mypy happy",
          "timestamp": "2024-03-14T09:53:39+01:00",
          "tree_id": "74312ac045a5af2e623615421171ff18f56d1b3f",
          "url": "https://github.com/equinor/ert/commit/d72d2fd9451caceebcb3ea7d48102af60266efc4"
        },
        "date": 1710406602291,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18978113344312603,
            "unit": "iter/sec",
            "range": "stddev: 0.052154846574549385",
            "extra": "mean: 5.269227672199998 sec\nrounds: 5"
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
          "id": "a08153e59ddcdc2f9462d4be101770933922ec67",
          "message": "Export qsub for pbs tests in testkomodo",
          "timestamp": "2024-03-14T12:33:53+01:00",
          "tree_id": "23611a98c892f175b5358f1d3a5e3102f42bfcc0",
          "url": "https://github.com/equinor/ert/commit/a08153e59ddcdc2f9462d4be101770933922ec67"
        },
        "date": 1710416227445,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1845912522069575,
            "unit": "iter/sec",
            "range": "stddev: 0.05707564749060804",
            "extra": "mean: 5.417374810799993 sec\nrounds: 5"
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
          "id": "d51d1f4c650b52cd94a7ee9b2593fb1052078d43",
          "message": "Store update log in experiment",
          "timestamp": "2024-03-14T15:16:26+01:00",
          "tree_id": "c281b3912ec372846a9e5d1c058c247f0cacc5b0",
          "url": "https://github.com/equinor/ert/commit/d51d1f4c650b52cd94a7ee9b2593fb1052078d43"
        },
        "date": 1710425979518,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18948107263123176,
            "unit": "iter/sec",
            "range": "stddev: 0.0494841611143443",
            "extra": "mean: 5.277571981800003 sec\nrounds: 5"
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
          "id": "7f31fd82640f16ea78100d7afbfa42e310cd74c3",
          "message": "Fix scheduler drivers not using queue config (#7448)\n\nFix scheduler create_driver not taking all queue options\r\n\r\nThis commit makes the scheduler actually use the missing queue_option \"LSF_QUEUE\" found in queue_config. It also adds some tests for the scheduler create_driver function.",
          "timestamp": "2024-03-14T15:32:03+01:00",
          "tree_id": "95743d83844b50d0cdbc018cbd5852b3932d0cb3",
          "url": "https://github.com/equinor/ert/commit/7f31fd82640f16ea78100d7afbfa42e310cd74c3"
        },
        "date": 1710426910609,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18666868663291536,
            "unit": "iter/sec",
            "range": "stddev: 0.05170536973854988",
            "extra": "mean: 5.357084886800021 sec\nrounds: 5"
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
          "id": "3834dd8831cde733d0ef02df7fe8c2f0af7c84f7",
          "message": "Let bsub retry on identified SSH failure",
          "timestamp": "2024-03-14T15:50:18+01:00",
          "tree_id": "8e4a36c0916ab7673cf3913b8f0114a0625354c3",
          "url": "https://github.com/equinor/ert/commit/3834dd8831cde733d0ef02df7fe8c2f0af7c84f7"
        },
        "date": 1710428005393,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18394529452195849,
            "unit": "iter/sec",
            "range": "stddev: 0.16900890856154802",
            "extra": "mean: 5.436398917400004 sec\nrounds: 5"
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
            "email": "berland@pvv.ntnu.no",
            "name": "Håvard Berland",
            "username": "berland"
          },
          "distinct": true,
          "id": "c57b5e15720eb656c2150a109b8b88b8229a09a5",
          "message": "Replace usage of a deprecated function in a test",
          "timestamp": "2024-03-14T15:52:23+01:00",
          "tree_id": "422b6baf73cc7c92298d0f836cb09a13cc75c233",
          "url": "https://github.com/equinor/ert/commit/c57b5e15720eb656c2150a109b8b88b8229a09a5"
        },
        "date": 1710428135208,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19072814826717488,
            "unit": "iter/sec",
            "range": "stddev: 0.030081174470503003",
            "extra": "mean: 5.243064587399994 sec\nrounds: 5"
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
          "id": "b12d8f6c5c8ffce80fabe1a2f68ad97251665d24",
          "message": "Remove fixture marks that has no effect\n\nSee https://docs.pytest.org/en/stable/deprecations.html#applying-a-mark-to-a-fixture-function",
          "timestamp": "2024-03-15T07:28:21+01:00",
          "tree_id": "c442d5f68f3f28dd4618881336abe2b156a1ff4b",
          "url": "https://github.com/equinor/ert/commit/b12d8f6c5c8ffce80fabe1a2f68ad97251665d24"
        },
        "date": 1710484300397,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18449156242546966,
            "unit": "iter/sec",
            "range": "stddev: 0.0718844461905878",
            "extra": "mean: 5.420302082399985 sec\nrounds: 5"
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
          "id": "01edea6b5a19132dc15a258670e23047ca949c76",
          "message": "Make PBS tests more tolerant for SIGTERM variants\n\nAt least on RHEL8 and Python 3.11 the return code from PBS has been observed to\nbe 128 + SIGTERM while at least on RHEL7/Python 3.8 it is 256 + SIGTERM.\n\nWe should accept both variants in tests.",
          "timestamp": "2024-03-15T07:29:08+01:00",
          "tree_id": "eb1e0ee320ce3d9e142ca166cd0ca6f7e58db089",
          "url": "https://github.com/equinor/ert/commit/01edea6b5a19132dc15a258670e23047ca949c76"
        },
        "date": 1710484340055,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18957470146407532,
            "unit": "iter/sec",
            "range": "stddev: 0.032078704209058714",
            "extra": "mean: 5.274965447800014 sec\nrounds: 5"
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
          "id": "26f4c2a86a6f4bd9c252c5149b407070a5a01893",
          "message": "Log failure to load ensemble",
          "timestamp": "2024-03-15T08:21:51+01:00",
          "tree_id": "2a56f6841b26148ff983c347fc8e71ecae4cfb5a",
          "url": "https://github.com/equinor/ert/commit/26f4c2a86a6f4bd9c252c5149b407070a5a01893"
        },
        "date": 1710487501100,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18850374373083603,
            "unit": "iter/sec",
            "range": "stddev: 0.030938167999382354",
            "extra": "mean: 5.304934428399986 sec\nrounds: 5"
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
          "id": "1e6d3fa50d7243289539ca1d10940e12adbf29eb",
          "message": "Add better feedback on no obs",
          "timestamp": "2024-03-15T08:59:23+01:00",
          "tree_id": "666b12318032b180003316c8005599e4637c0721",
          "url": "https://github.com/equinor/ert/commit/1e6d3fa50d7243289539ca1d10940e12adbf29eb"
        },
        "date": 1710489743219,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19250533091912939,
            "unit": "iter/sec",
            "range": "stddev: 0.021387836259504463",
            "extra": "mean: 5.194661338599997 sec\nrounds: 5"
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
          "id": "bfbbe09cb479904407c8e5c05ff4650cc6b2fdaa",
          "message": "Have OpenPBS driver use qstat -w option\n\n* Have OpenPBS driver use qstat -w option",
          "timestamp": "2024-03-15T09:10:55+01:00",
          "tree_id": "c660b1cd2e5e4a2541c239d95bb8fec68f88f3ca",
          "url": "https://github.com/equinor/ert/commit/bfbbe09cb479904407c8e5c05ff4650cc6b2fdaa"
        },
        "date": 1710490444897,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19060157249635876,
            "unit": "iter/sec",
            "range": "stddev: 0.033497533072988556",
            "extra": "mean: 5.246546431400003 sec\nrounds: 5"
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
            "email": "sondreso@users.noreply.github.com",
            "name": "Sondre Sortland",
            "username": "sondreso"
          },
          "distinct": true,
          "id": "ffae27e612339dc772ea7f9e6f81478b8da90262",
          "message": "Update contributing.md",
          "timestamp": "2024-03-15T09:22:11+01:00",
          "tree_id": "7d15e168eae4a3b32bbc397d95507207a0a77862",
          "url": "https://github.com/equinor/ert/commit/ffae27e612339dc772ea7f9e6f81478b8da90262"
        },
        "date": 1710491118891,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19040474037563954,
            "unit": "iter/sec",
            "range": "stddev: 0.03464584436605219",
            "extra": "mean: 5.2519700824000095 sec\nrounds: 5"
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
          "id": "fd5a365cd5c61e3ffe618d2d34b5da96d2aca23b",
          "message": "Rename `case` to `ensemble`\n\nReplace current_case and target_case with ensemble\r\n\r\nRemove requirement that ES run from CLI needs --target-case,\r\nbecause it is not required in the GUI.\r\n\r\nDeprecate current-case and target-case in CLI",
          "timestamp": "2024-03-15T11:40:08+01:00",
          "tree_id": "574b43dd34c3e445d13c0d82b8ef3d62c8893c01",
          "url": "https://github.com/equinor/ert/commit/fd5a365cd5c61e3ffe618d2d34b5da96d2aca23b"
        },
        "date": 1710499394825,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18783823355908133,
            "unit": "iter/sec",
            "range": "stddev: 0.129820003242732",
            "extra": "mean: 5.323729791599999 sec\nrounds: 5"
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
          "id": "a195daa86b002db5e83cf2a90ebdb5036a80c544",
          "message": "Fix LSF drivers ability to send to non-default queue\n\nThe Python LSF driver had a bug in a regexp causing it\nto crash if one tried to use a queue other than the default.",
          "timestamp": "2024-03-15T12:53:36+01:00",
          "tree_id": "3787e47856b7dbcd4b41e91c8cf41655425899c8",
          "url": "https://github.com/equinor/ert/commit/a195daa86b002db5e83cf2a90ebdb5036a80c544"
        },
        "date": 1710503799141,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18915112702360518,
            "unit": "iter/sec",
            "range": "stddev: 0.021877093235079602",
            "extra": "mean: 5.2867779100000005 sec\nrounds: 5"
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
          "id": "77e3139642d5b261764aa21f5aa4cf876e1bf7e7",
          "message": "Rewrite gui tests to have a clean app/storage for each test",
          "timestamp": "2024-03-15T12:57:12+01:00",
          "tree_id": "659c10c1988c04344e593c44319d71f3a03cbcbd",
          "url": "https://github.com/equinor/ert/commit/77e3139642d5b261764aa21f5aa4cf876e1bf7e7"
        },
        "date": 1710504025473,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1875578509976704,
            "unit": "iter/sec",
            "range": "stddev: 0.035833948220759214",
            "extra": "mean: 5.331688301399981 sec\nrounds: 5"
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
          "id": "cf6005e3eb9abf05a03380e8bfe2a8db7d70b340",
          "message": "Stop failing on upload error",
          "timestamp": "2024-03-15T13:54:57+01:00",
          "tree_id": "0a3029b87dc8d64f8265ee696fe0394ad611259d",
          "url": "https://github.com/equinor/ert/commit/cf6005e3eb9abf05a03380e8bfe2a8db7d70b340"
        },
        "date": 1710507496087,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19128430113086228,
            "unit": "iter/sec",
            "range": "stddev: 0.019113974030628315",
            "extra": "mean: 5.227820548200009 sec\nrounds: 5"
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
          "id": "819de9417a134f4fa6ba588e1d1c3b9d391b5166",
          "message": "Fix integration tests with real LSF cluster\n\nThe LSF driver writes its job script to disk and sends that script path\nto the LSF cluster through bsub. If the job script is not on a shared\ndisk the job will fail silently.",
          "timestamp": "2024-03-15T15:03:57+01:00",
          "tree_id": "ca5b095e7e69492d8300fb3268c6d6036846afc8",
          "url": "https://github.com/equinor/ert/commit/819de9417a134f4fa6ba588e1d1c3b9d391b5166"
        },
        "date": 1710511624623,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18992629412739429,
            "unit": "iter/sec",
            "range": "stddev: 0.04214575572585793",
            "extra": "mean: 5.265200400999999 sec\nrounds: 5"
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
          "id": "037606a9a8a5e3b09f41f0ef5b08da7c6b055471",
          "message": "Add handling for unknown module",
          "timestamp": "2024-03-15T20:02:37+01:00",
          "tree_id": "afd238f1a58aca2a1afda994b56634893303b6f4",
          "url": "https://github.com/equinor/ert/commit/037606a9a8a5e3b09f41f0ef5b08da7c6b055471"
        },
        "date": 1710529549733,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18415144575859044,
            "unit": "iter/sec",
            "range": "stddev: 0.03870528304095054",
            "extra": "mean: 5.430313055000011 sec\nrounds: 5"
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
          "id": "0d1b285e0b88c70770ffad820cec1ccf02fe10a0",
          "message": "Remove unused file exists exception",
          "timestamp": "2024-03-18T08:10:09+01:00",
          "tree_id": "fb1c0e762b9677d042de19b846d3a45e79fac5a4",
          "url": "https://github.com/equinor/ert/commit/0d1b285e0b88c70770ffad820cec1ccf02fe10a0"
        },
        "date": 1710746017134,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.18929464354066275,
            "unit": "iter/sec",
            "range": "stddev: 0.02291182619699376",
            "extra": "mean: 5.282769661600002 sec\nrounds: 5"
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
          "id": "13e45707b2a232757ada36023dca904ee5d22c2e",
          "message": "Allow cleanup rm commands to fail\n\nThese rm commands operate on a NFS share, and removal on NFS\nshares occasionally fail due to some lock files or similar being\nhard to remove.",
          "timestamp": "2024-03-18T13:57:18+01:00",
          "tree_id": "fe82332448c34fa2783e93708553070aea89bf24",
          "url": "https://github.com/equinor/ert/commit/13e45707b2a232757ada36023dca904ee5d22c2e"
        },
        "date": 1710766903418,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.1859576522602376,
            "unit": "iter/sec",
            "range": "stddev: 0.13081816100983812",
            "extra": "mean: 5.377568429400014 sec\nrounds: 5"
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
          "id": "e4d25a12f028144bc87a28906fd1f413c277d8ed",
          "message": "Ensure no overflow in parameter_example test",
          "timestamp": "2024-03-19T09:25:02+01:00",
          "tree_id": "1a2bbafcfd935959ab0d40ec619eee7fb41d75d1",
          "url": "https://github.com/equinor/ert/commit/e4d25a12f028144bc87a28906fd1f413c277d8ed"
        },
        "date": 1710836909982,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 0.19106880284409453,
            "unit": "iter/sec",
            "range": "stddev: 0.0360058787994941",
            "extra": "mean: 5.233716782199997 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}