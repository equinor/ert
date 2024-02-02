window.BENCHMARK_DATA = {
  "lastUpdate": 1706887684721,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "a90b11a0ecc25aa73ea8f7549d5aeaf78002ccae",
          "message": "Remove mention of opencensus.\n\nOpencensus was replaced with Opentelemetry and there is no further reason to exclude `opencensus.ext.azure.common.transport` records when capturing logs.",
          "timestamp": "2024-01-29T15:59:00+02:00",
          "tree_id": "8447805122c44d917fa1f390b3d0886226d974e5",
          "url": "https://github.com/equinor/ert/commit/a90b11a0ecc25aa73ea8f7549d5aeaf78002ccae"
        },
        "date": 1706536888740,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.700251480672506,
            "unit": "iter/sec",
            "range": "stddev: 0.002496472154968146",
            "extra": "mean: 149.2481294000072 msec\nrounds: 5"
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
          "id": "0c4a70cd12abb2b19d743616163131c067fbe4bf",
          "message": "Check scheduler loop is still running before killing all the jobs.",
          "timestamp": "2024-01-29T16:53:45+02:00",
          "tree_id": "d8e116f61749fbcc93b91e30542909da4ba90606",
          "url": "https://github.com/equinor/ert/commit/0c4a70cd12abb2b19d743616163131c067fbe4bf"
        },
        "date": 1706540177244,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.537034919594398,
            "unit": "iter/sec",
            "range": "stddev: 0.002100702794487692",
            "extra": "mean: 152.9745538000043 msec\nrounds: 5"
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
          "id": "6b784ef056a9d3f09b284aa3557f67f238386c50",
          "message": "Move experiment creation",
          "timestamp": "2024-01-30T10:19:02+01:00",
          "tree_id": "4ed139c04c9d451cef3c4a3da638eb3a2c8c1001",
          "url": "https://github.com/equinor/ert/commit/6b784ef056a9d3f09b284aa3557f67f238386c50"
        },
        "date": 1706606502221,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.401245622994807,
            "unit": "iter/sec",
            "range": "stddev: 0.006792502901449424",
            "extra": "mean: 156.21959519999677 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "1a5b6a6be9cb7fda0b99a12088b25bf5a548f2c0",
          "message": "Do not close main asyncio event loop of ert in scheduler",
          "timestamp": "2024-01-30T10:32:23+01:00",
          "tree_id": "d0a5746c30783e8c932b5e4bad658aeab546014e",
          "url": "https://github.com/equinor/ert/commit/1a5b6a6be9cb7fda0b99a12088b25bf5a548f2c0"
        },
        "date": 1706607303802,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7133615423858055,
            "unit": "iter/sec",
            "range": "stddev: 0.002594944457024593",
            "extra": "mean: 148.95667300000923 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jholba@equinor.com",
            "name": "Jon Holba"
          },
          "committer": {
            "email": "jon.holba@gmail.com",
            "name": "Jon Holba",
            "username": "JHolba"
          },
          "distinct": true,
          "id": "64ad91a55b9d470dbc8e0681669057ab312cb971",
          "message": "Add documentation for queue behavior on early exit",
          "timestamp": "2024-01-30T10:32:37+01:00",
          "tree_id": "41872aab1ebb263dd1974a7067c4a3891cef6d3e",
          "url": "https://github.com/equinor/ert/commit/64ad91a55b9d470dbc8e0681669057ab312cb971"
        },
        "date": 1706607312535,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.703020228756857,
            "unit": "iter/sec",
            "range": "stddev: 0.0026208204216903873",
            "extra": "mean: 149.18648099999245 msec\nrounds: 5"
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
          "id": "055e4fd0ec724caa35b178591bd1342fdc456089",
          "message": "Add TorqueDriver",
          "timestamp": "2024-01-30T10:39:32+01:00",
          "tree_id": "1253392d364d0f9a7b262dd685f5207bceadc95b",
          "url": "https://github.com/equinor/ert/commit/055e4fd0ec724caa35b178591bd1342fdc456089"
        },
        "date": 1706607724829,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.065828346677965,
            "unit": "iter/sec",
            "range": "stddev: 0.0317900236936553",
            "extra": "mean: 164.85794566667286 msec\nrounds: 6"
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
          "id": "9f74a17d64076f5e4164b3bd323ce9bf5b5f2fa1",
          "message": "Replace observation_config with observations",
          "timestamp": "2024-01-30T11:21:04+01:00",
          "tree_id": "fdf746e0a9de332a92fee6f7adc514441c7fc723",
          "url": "https://github.com/equinor/ert/commit/9f74a17d64076f5e4164b3bd323ce9bf5b5f2fa1"
        },
        "date": 1706610213865,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.636720676853125,
            "unit": "iter/sec",
            "range": "stddev: 0.0008689042703591679",
            "extra": "mean: 150.67682499998796 msec\nrounds: 5"
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
          "id": "0e6a770eacda8180aa2f5ba2196b98985ebf78f0",
          "message": "Fix a bug in migration",
          "timestamp": "2024-01-30T11:27:40+01:00",
          "tree_id": "e19f7a4799bc89c7001e40fc05927ca4cc903519",
          "url": "https://github.com/equinor/ert/commit/0e6a770eacda8180aa2f5ba2196b98985ebf78f0"
        },
        "date": 1706610613261,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.177102844688486,
            "unit": "iter/sec",
            "range": "stddev: 0.031197179684927155",
            "extra": "mean: 161.88819016666875 msec\nrounds: 6"
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
          "id": "48d428aaedbb0fd64dc5653e663b05d7421b0c95",
          "message": "Remove unused ignore_current from CaseSelector",
          "timestamp": "2024-01-30T11:53:34+01:00",
          "tree_id": "5c61566e121e94572b9bf02ef177daf02ec567e4",
          "url": "https://github.com/equinor/ert/commit/48d428aaedbb0fd64dc5653e663b05d7421b0c95"
        },
        "date": 1706612177458,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.69648602966981,
            "unit": "iter/sec",
            "range": "stddev: 0.002181149095666006",
            "extra": "mean: 149.33205200001112 msec\nrounds: 5"
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
          "id": "fd17a0fd58cebebfbe5c812f2a2f2c58c2295dbe",
          "message": "Remove libres from remaining endpoints",
          "timestamp": "2024-01-30T14:18:40+01:00",
          "tree_id": "8ea54e29752f0af31f80154069993af46da363c1",
          "url": "https://github.com/equinor/ert/commit/fd17a0fd58cebebfbe5c812f2a2f2c58c2295dbe"
        },
        "date": 1706620884741,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.974379256917177,
            "unit": "iter/sec",
            "range": "stddev: 0.03417923555673988",
            "extra": "mean: 167.3814059999946 msec\nrounds: 6"
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
          "id": "36bc137c0a41e3f6c2062c8981ce80009b0fd096",
          "message": "Remove some more resdata",
          "timestamp": "2024-01-31T08:37:08+01:00",
          "tree_id": "8df7e4dd389180b44ddbb142953a6ef92bcd3a80",
          "url": "https://github.com/equinor/ert/commit/36bc137c0a41e3f6c2062c8981ce80009b0fd096"
        },
        "date": 1706686785840,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.574117986747657,
            "unit": "iter/sec",
            "range": "stddev: 0.0019218401960360968",
            "extra": "mean: 152.1116599999933 msec\nrounds: 6"
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
          "id": "2c65ef14583ba170768d51ffef7a1e9d589b2897",
          "message": "Speed up tests",
          "timestamp": "2024-01-31T08:41:51+01:00",
          "tree_id": "df7ac73e19f2837d5b1d9e67bc3b6c25333238b1",
          "url": "https://github.com/equinor/ert/commit/2c65ef14583ba170768d51ffef7a1e9d589b2897"
        },
        "date": 1706687069446,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.507933088000795,
            "unit": "iter/sec",
            "range": "stddev: 0.0025920491837348974",
            "extra": "mean: 153.65861733332528 msec\nrounds: 6"
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
          "id": "8c843f443b2d8ae2d560f52047ed2e4f1fab3267",
          "message": "Speed up local_ensemble",
          "timestamp": "2024-01-31T09:29:20+01:00",
          "tree_id": "26ce5b334fcb922532e7e4e8cd13e2049814f5ec",
          "url": "https://github.com/equinor/ert/commit/8c843f443b2d8ae2d560f52047ed2e4f1fab3267"
        },
        "date": 1706689914449,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7346437511849135,
            "unit": "iter/sec",
            "range": "stddev: 0.0023307173258653967",
            "extra": "mean: 148.4859536666742 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zom@be-lx136139.be.statoil.no",
            "name": "Zohar Malamant (EDT DSD SD2)"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "027c80d714d9c3c17eda5a8688c9039c71bb3eea",
          "message": "Rename TorqueDriver to OpenPBSDriver\n\nThis more accurately represents the HPC system that we are targetting.",
          "timestamp": "2024-01-31T10:48:59+01:00",
          "tree_id": "6f0ee48d512543fa07e728ba39472cc0c98581d1",
          "url": "https://github.com/equinor/ert/commit/027c80d714d9c3c17eda5a8688c9039c71bb3eea"
        },
        "date": 1706694691566,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.738133648023847,
            "unit": "iter/sec",
            "range": "stddev: 0.0023722570449287573",
            "extra": "mean: 148.40904800000203 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "zom@equinor.com",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "committer": {
            "email": "git@wah.pink",
            "name": "Zohar Malamant",
            "username": "pinkwah"
          },
          "distinct": true,
          "id": "74975709078a89cb14967de543d1a3906b6c8697",
          "message": "Fix mypy in Python 3.12",
          "timestamp": "2024-01-31T14:11:15+01:00",
          "tree_id": "ab167db09deb2dd63d3b6355a45d5efba7e88c07",
          "url": "https://github.com/equinor/ert/commit/74975709078a89cb14967de543d1a3906b6c8697"
        },
        "date": 1706706872780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6936604589458835,
            "unit": "iter/sec",
            "range": "stddev: 0.0016334656299865284",
            "extra": "mean: 149.39508899999984 msec\nrounds: 6"
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
          "id": "d766260b17d9e2eba2c9f6589604e1ef9fae549b",
          "message": "Add support for Python 3.12",
          "timestamp": "2024-01-31T16:19:04+01:00",
          "tree_id": "b2105b74747c40f48d9704408e674b5391496653",
          "url": "https://github.com/equinor/ert/commit/d766260b17d9e2eba2c9f6589604e1ef9fae549b"
        },
        "date": 1706714512763,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.697412956587076,
            "unit": "iter/sec",
            "range": "stddev: 0.002171697843102207",
            "extra": "mean: 149.31138433333047 msec\nrounds: 6"
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
          "id": "47042759efb73ed7da1c02b92c0536f603d350ea",
          "message": "Use .size to get number of observations",
          "timestamp": "2024-01-31T19:26:21+01:00",
          "tree_id": "84a77e4341bbcd337dcac113a889c3329cb3c081",
          "url": "https://github.com/equinor/ert/commit/47042759efb73ed7da1c02b92c0536f603d350ea"
        },
        "date": 1706725768715,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.473572381438012,
            "unit": "iter/sec",
            "range": "stddev: 0.002892088524634427",
            "extra": "mean: 154.474213166651 msec\nrounds: 6"
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
          "id": "66412cfcbfe44b0c45f7a6343e5ae7ba70f139b8",
          "message": "simplify testing of create_runpath",
          "timestamp": "2024-02-01T08:09:50+01:00",
          "tree_id": "8b2a68616f0f71d2fc897eb8ea45c12af4d9f63d",
          "url": "https://github.com/equinor/ert/commit/66412cfcbfe44b0c45f7a6343e5ae7ba70f139b8"
        },
        "date": 1706771541882,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.726583341556464,
            "unit": "iter/sec",
            "range": "stddev: 0.002005787025090977",
            "extra": "mean: 148.6638831666672 msec\nrounds: 6"
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
          "id": "10241d458b87afba254a490bc93db6ccda2be3e8",
          "message": "Fix typo in typing",
          "timestamp": "2024-02-01T12:30:19+01:00",
          "tree_id": "39677a4904ccec09e83e6ca0771c17a346b2fc24",
          "url": "https://github.com/equinor/ert/commit/10241d458b87afba254a490bc93db6ccda2be3e8"
        },
        "date": 1706787171056,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.644174761968087,
            "unit": "iter/sec",
            "range": "stddev: 0.0025148335033196353",
            "extra": "mean: 150.50778099999698 msec\nrounds: 6"
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
          "id": "6b7eb02053a32459730e57a6a6dc237aa69fbc7d",
          "message": "Update snake_oil.ert",
          "timestamp": "2024-02-01T15:25:02+01:00",
          "tree_id": "c442d9b57705a6df6bf43e69deaf9fa112ac6533",
          "url": "https://github.com/equinor/ert/commit/6b7eb02053a32459730e57a6a6dc237aa69fbc7d"
        },
        "date": 1706797665135,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.557655502283213,
            "unit": "iter/sec",
            "range": "stddev: 0.003796424448958643",
            "extra": "mean: 152.4935245000023 msec\nrounds: 6"
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
          "id": "3cead86221d5a029ecf23d78a230bfe939ace0cf",
          "message": "Test semeio with python 3.11\n\nsemeio doesn't support 3.12 yet",
          "timestamp": "2024-02-02T09:19:36+01:00",
          "tree_id": "4cfe7772c8cc13b80d8fe59632aae7da76195a00",
          "url": "https://github.com/equinor/ert/commit/3cead86221d5a029ecf23d78a230bfe939ace0cf"
        },
        "date": 1706862126647,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.713343589860566,
            "unit": "iter/sec",
            "range": "stddev: 0.0025867370599206153",
            "extra": "mean: 148.95707133332792 msec\nrounds: 6"
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
          "id": "949aa1aa6ce6e44720983d89c7df1c2a941474c1",
          "message": "Change some local storage tests to stateful",
          "timestamp": "2024-02-02T09:37:29+01:00",
          "tree_id": "65681f94b0db141f8e930929a2c9b4b2788027a7",
          "url": "https://github.com/equinor/ert/commit/949aa1aa6ce6e44720983d89c7df1c2a941474c1"
        },
        "date": 1706863200303,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.28813670414201,
            "unit": "iter/sec",
            "range": "stddev: 0.015588627794159626",
            "extra": "mean: 159.02962150000613 msec\nrounds: 6"
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
          "id": "98a43cc08f062fe1e13ceea66dbed199455dd77a",
          "message": "Speed up local_experiment",
          "timestamp": "2024-02-02T10:06:29+01:00",
          "tree_id": "4ef7b363ebf50a2db88d48d6f6d73f56e2438992",
          "url": "https://github.com/equinor/ert/commit/98a43cc08f062fe1e13ceea66dbed199455dd77a"
        },
        "date": 1706864939390,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.62000198580201,
            "unit": "iter/sec",
            "range": "stddev: 0.008651133531991987",
            "extra": "mean: 151.05735649999966 msec\nrounds: 6"
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
          "id": "848c2be777566120780247eda30c05df98cf299a",
          "message": "Fix flakiness for integration_test and gui unit_tests (#7077)\n\n* Fix flakiness unit_tests/gui timeout\r\n\r\nImproves flakiness of unit_tests/gui/test_main_window::test_that_run_dialog_can_be_closed_after_used_to_open_plots by increasing timeout time for done_button to be enabled. This also applies to unit_tests/gui/simulation/test_run_dialog.py::test_that_run_dialog_can_be_closed_while_file_plot_is_open. In a different fixture, the timeout for running the experiment is set to 200.000, while here it was only 20.000. This commit increases this timeout to 200.000.\r\n\r\n* Fix flaky unit test gui plotapi sharing storage\r\n\r\nWhen running some of the gui unit tests in parallell, they would share storage. This was very prevalent for the test unit_tests/gui/test_main_window.py::test_that_gui_plotter_works_when_no_data, where the PlotApi would return data from the other tests (which running esmda). This is fixed in this commit by mocking the return value from PlotApi.\r\n\r\n* Increase timeout on flaky test\r\n\r\nIncrease timeout to 60s for integration_tests/scheduler/test_integration_local_driver.py::test_subprocesses_live_on_after_ert_dies.",
          "timestamp": "2024-02-02T11:17:11Z",
          "tree_id": "85861670ac39390ece7c77e8f3567b00209f4bae",
          "url": "https://github.com/equinor/ert/commit/848c2be777566120780247eda30c05df98cf299a"
        },
        "date": 1706872782262,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.933083656043518,
            "unit": "iter/sec",
            "range": "stddev: 0.0024126265526539107",
            "extra": "mean: 144.23596333332966 msec\nrounds: 6"
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
          "id": "65404c72083c34a367c73f92258b89f90a2c4977",
          "message": "Add field and observations to state storage test",
          "timestamp": "2024-02-02T12:56:59+01:00",
          "tree_id": "e0bd5c0a678a599b467f15b90e70474d0b0dcc78",
          "url": "https://github.com/equinor/ert/commit/65404c72083c34a367c73f92258b89f90a2c4977"
        },
        "date": 1706875166764,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.875339247858842,
            "unit": "iter/sec",
            "range": "stddev: 0.00281461608116104",
            "extra": "mean: 145.4473683333409 msec\nrounds: 6"
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
          "id": "77da2a3c4ea079cff5ff680550fcaa98d1b1feb4",
          "message": "Ensure good error message for no summary data",
          "timestamp": "2024-02-02T12:57:13+01:00",
          "tree_id": "3fea223d6e50bc550acd30b78176e3f2ac5a9aa0",
          "url": "https://github.com/equinor/ert/commit/77da2a3c4ea079cff5ff680550fcaa98d1b1feb4"
        },
        "date": 1706875188459,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.846390620235092,
            "unit": "iter/sec",
            "range": "stddev: 0.001675542657263967",
            "extra": "mean: 146.06236416666243 msec\nrounds: 6"
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
          "id": "40b4ecd6d2e6df582556918076f094f8dabee6e3",
          "message": "Remove unused include",
          "timestamp": "2024-02-02T13:08:45+01:00",
          "tree_id": "c2a1972ad8001876c6f55688de853ece589b123b",
          "url": "https://github.com/equinor/ert/commit/40b4ecd6d2e6df582556918076f094f8dabee6e3"
        },
        "date": 1706875916452,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.618200014301109,
            "unit": "iter/sec",
            "range": "stddev: 0.008334418124477728",
            "extra": "mean: 151.09848566666528 msec\nrounds: 6"
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
          "id": "f694de0778a20b99bf84d6ba69246f38cb38a132",
          "message": "Increase wait time for test_integration_local_driver.py",
          "timestamp": "2024-02-02T15:41:54+01:00",
          "tree_id": "a0e8327ac6d13322154c0a06e989f26eaa697c06",
          "url": "https://github.com/equinor/ert/commit/f694de0778a20b99bf84d6ba69246f38cb38a132"
        },
        "date": 1706885064809,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.811914299687678,
            "unit": "iter/sec",
            "range": "stddev: 0.003021839937142708",
            "extra": "mean: 146.80161199999966 msec\nrounds: 6"
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
          "id": "e87e86a9ba6f6db063a604fbe5e65bb05042d768",
          "message": "Don't keep output in JobQueue Torque\n\nThe `-k` option in `qsub` counter-intuitively means which things to\ndiscard. `-koe` means to discard both stdout and stderr of the job,\nwhich is what we want.",
          "timestamp": "2024-02-02T16:20:06+01:00",
          "tree_id": "cff67d408db3faf9e5822887cbbe3aadfb30a939",
          "url": "https://github.com/equinor/ert/commit/e87e86a9ba6f6db063a604fbe5e65bb05042d768"
        },
        "date": 1706887353499,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.866106814617098,
            "unit": "iter/sec",
            "range": "stddev: 0.002601068373001795",
            "extra": "mean: 145.6429424999802 msec\nrounds: 6"
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
          "id": "cec92030df30c7728df049c9970057bb214d56cc",
          "message": "Allow TORQUE queue with --enable-scheduler",
          "timestamp": "2024-02-02T16:25:21+01:00",
          "tree_id": "e4082e750a4ba279c82b4bf7b9c95905475e9bc4",
          "url": "https://github.com/equinor/ert/commit/cec92030df30c7728df049c9970057bb214d56cc"
        },
        "date": 1706887683904,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.752702724528988,
            "unit": "iter/sec",
            "range": "stddev: 0.0010233863986952928",
            "extra": "mean: 148.08885283333004 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}