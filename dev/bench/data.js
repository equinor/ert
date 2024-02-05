window.BENCHMARK_DATA = {
  "lastUpdate": 1707137437293,
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
          "id": "2a5506026c0d70ae02fce2f02effc00c42540265",
          "message": "Make updating work with failed realizations",
          "timestamp": "2024-02-02T17:19:56+01:00",
          "tree_id": "f9450670a283ae85fc3bc429f7f2848a9a8da64b",
          "url": "https://github.com/equinor/ert/commit/2a5506026c0d70ae02fce2f02effc00c42540265"
        },
        "date": 1706890948970,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.823966013391923,
            "unit": "iter/sec",
            "range": "stddev: 0.002692077930838845",
            "extra": "mean: 146.54234766666718 msec\nrounds: 6"
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
          "id": "22b5cd483b6df7c25bc10f700679783a39b9c9ba",
          "message": "Fix migration from 8.0.12 to 8.4.x",
          "timestamp": "2024-02-02T17:32:13+01:00",
          "tree_id": "9168acabc0bd1af09684737582c6b816ab4fba24",
          "url": "https://github.com/equinor/ert/commit/22b5cd483b6df7c25bc10f700679783a39b9c9ba"
        },
        "date": 1706891686965,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.884411902538182,
            "unit": "iter/sec",
            "range": "stddev: 0.0026243131846327527",
            "extra": "mean: 145.25568983333415 msec\nrounds: 6"
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
          "id": "b35134a8693e851ee67ca8bc7d8ffefeb413c3e0",
          "message": "Pin hypothesis version",
          "timestamp": "2024-02-05T10:01:04+01:00",
          "tree_id": "bbbdd4babff83ed4398cb318f62c44dabc40ad85",
          "url": "https://github.com/equinor/ert/commit/b35134a8693e851ee67ca8bc7d8ffefeb413c3e0"
        },
        "date": 1707123830297,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 7.063719401639363,
            "unit": "iter/sec",
            "range": "stddev: 0.0033657036562662708",
            "extra": "mean: 141.56847733333203 msec\nrounds: 6"
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
          "id": "faf029d80889a2c2a763f9cb16d749f885800c86",
          "message": "Fix test data generation failing\n\nhypothesis sometimes fails a health check with\ntoo much test data removed due to failed assumption.\n\nThe failing assumption is changed to an assignment",
          "timestamp": "2024-02-05T10:05:55+01:00",
          "tree_id": "35f7bb0dc7f7c0c04d4a83b14475366ffa22cc0b",
          "url": "https://github.com/equinor/ert/commit/faf029d80889a2c2a763f9cb16d749f885800c86"
        },
        "date": 1707124159461,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 7.089229752918906,
            "unit": "iter/sec",
            "range": "stddev: 0.004654904784152193",
            "extra": "mean: 141.05904799999772 msec\nrounds: 6"
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
          "id": "a1eb1b922b2d1c84cba9271282b783dc628ae7dd",
          "message": "Unpin hypothesis version",
          "timestamp": "2024-02-05T10:42:17+01:00",
          "tree_id": "778fa9459c0358f1ab0a1af35d1085d6e25e984c",
          "url": "https://github.com/equinor/ert/commit/a1eb1b922b2d1c84cba9271282b783dc628ae7dd"
        },
        "date": 1707126294145,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.443329352082363,
            "unit": "iter/sec",
            "range": "stddev: 0.032325512951918545",
            "extra": "mean: 155.19926816666896 msec\nrounds: 6"
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
          "id": "e35e7e07b39534ac3ed83085a87a970b633dcc70",
          "message": "Update codecov uploader",
          "timestamp": "2024-02-05T11:01:42+01:00",
          "tree_id": "a8e76883b5e07552854c458080376f601a0bb5bd",
          "url": "https://github.com/equinor/ert/commit/e35e7e07b39534ac3ed83085a87a970b633dcc70"
        },
        "date": 1707127469304,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 5.690313404812423,
            "unit": "iter/sec",
            "range": "stddev: 0.08454328250779851",
            "extra": "mean: 175.7372448333475 msec\nrounds: 6"
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
          "id": "db56a48697a130591edb4688ca52992d589972d9",
          "message": "Replace some tests with an integration test\n\nAvoids relying on details. Testing of storage api is\ndelegated to the StorageTest RuleBasedStateMachine.",
          "timestamp": "2024-02-05T13:47:20+01:00",
          "tree_id": "f735293ba9f5f6b8278699b574293cb2d4159e50",
          "url": "https://github.com/equinor/ert/commit/db56a48697a130591edb4688ca52992d589972d9"
        },
        "date": 1707137436454,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.467002087358255,
            "unit": "iter/sec",
            "range": "stddev: 0.03205177609819873",
            "extra": "mean: 154.63115466667432 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}