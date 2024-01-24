window.BENCHMARK_DATA = {
  "lastUpdate": 1706096663122,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "id": "dd3a01de078e91dbda15e9d39c68f22a61f1ffb6",
          "message": "Move loggin msg to Scheduler when cancelling",
          "timestamp": "2024-01-11T12:04:12+01:00",
          "tree_id": "f9f338d7d41e8cae64e75bbd375423096c59ab75",
          "url": "https://github.com/equinor/ert/commit/dd3a01de078e91dbda15e9d39c68f22a61f1ffb6"
        },
        "date": 1704971210348,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.038237616729065,
            "unit": "iter/sec",
            "range": "stddev: 0.036077801837768375",
            "extra": "mean: 165.6112368333235 msec\nrounds: 6"
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
          "id": "f687bbad1a029a1be675885817ab29d618ebf9db",
          "message": "Cancel correctly tasks from sync context in Scheduler\n\nWhen stopping the executing from ee, which runs in another thread, we need to use the correct loop\nwhen cancelling the job tasks. Further, we just signal to cancel therefore we don't need to await\nfor the tasks to finish. This is handled in the Scheduler.execute - asyncio.gather.\n\nThere two functions (kill_all_jobs and cancel_all_jobs) to cancel the tasks in the Scheduler. kill_all_jobs is meant to be used from sync context.",
          "timestamp": "2024-01-12T13:44:47+01:00",
          "tree_id": "62bb8cfbb5d9d3920511e4b73be0bbbd3f9f86e1",
          "url": "https://github.com/equinor/ert/commit/f687bbad1a029a1be675885817ab29d618ebf9db"
        },
        "date": 1705063650251,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.614316614231362,
            "unit": "iter/sec",
            "range": "stddev: 0.0017013985557833187",
            "extra": "mean: 151.18719866666197 msec\nrounds: 6"
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
          "id": "0f93287ada2351cb0dfd3a105e4346b6ea01ab94",
          "message": "Fix bug where other parameters than GenKw would fail ert",
          "timestamp": "2024-01-12T14:51:05+01:00",
          "tree_id": "629778162d73fd748100d7d98e16c9f8929748cc",
          "url": "https://github.com/equinor/ert/commit/0f93287ada2351cb0dfd3a105e4346b6ea01ab94"
        },
        "date": 1705067622305,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.598519815415804,
            "unit": "iter/sec",
            "range": "stddev: 0.002386032979786403",
            "extra": "mean: 151.54913950000548 msec\nrounds: 6"
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
          "id": "bb5fa0d66c888fa015ea61d3d4f3d773dabbf56f",
          "message": "Use rangestring_to_list instead of IntVector",
          "timestamp": "2024-01-12T15:23:26+01:00",
          "tree_id": "c65cc4e51c0ebdfe6f97e45e7009b60d45bb6c1a",
          "url": "https://github.com/equinor/ert/commit/bb5fa0d66c888fa015ea61d3d4f3d773dabbf56f"
        },
        "date": 1705069561661,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.60546775652208,
            "unit": "iter/sec",
            "range": "stddev: 0.002477576643803553",
            "extra": "mean: 151.38973300000202 msec\nrounds: 6"
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
          "id": "17426909b0c57d7beac907b8c8a9c8db24fe9481",
          "message": "Use resfo instead of eclsum for run_ecl\n\nThis avoids critical failures from resdata propagating to the user.",
          "timestamp": "2024-01-12T15:23:11+01:00",
          "tree_id": "060e9ca88dcaf257d6cafce8104f29bbe60e80ca",
          "url": "https://github.com/equinor/ert/commit/17426909b0c57d7beac907b8c8a9c8db24fe9481"
        },
        "date": 1705069562621,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.641475035299548,
            "unit": "iter/sec",
            "range": "stddev: 0.002555100261530855",
            "extra": "mean: 150.56896166664538 msec\nrounds: 6"
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
          "id": "3db8f4efe0313326df270abac86fe9e416823acf",
          "message": "Fix ecl_run eclipse tests",
          "timestamp": "2024-01-14T12:07:19+01:00",
          "tree_id": "16fb7bba44a35ea773c34d45b88052cc4efb0281",
          "url": "https://github.com/equinor/ert/commit/3db8f4efe0313326df270abac86fe9e416823acf"
        },
        "date": 1705230590237,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.621007945745749,
            "unit": "iter/sec",
            "range": "stddev: 0.0030571618030448898",
            "extra": "mean: 151.03440566666868 msec\nrounds: 6"
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
          "id": "7ef6fd6c628bb10954485d6368affb0e871b37e5",
          "message": "Extend license retry to more error messages",
          "timestamp": "2024-01-15T09:02:21+01:00",
          "tree_id": "79f484b2302c8ed7310a2871fa91207acac64d47",
          "url": "https://github.com/equinor/ert/commit/7ef6fd6c628bb10954485d6368affb0e871b37e5"
        },
        "date": 1705305885465,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.57400713181161,
            "unit": "iter/sec",
            "range": "stddev: 0.003128542102445433",
            "extra": "mean: 152.11422500000063 msec\nrounds: 6"
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
          "id": "adf59e896075536f18cee1d6492cf56ded2539e6",
          "message": "Set lower-bound of iterative_ensemble_smoother",
          "timestamp": "2024-01-17T08:57:37+01:00",
          "tree_id": "f3c7a76e7dfbabe3fe4cc8c28d12d368f35cad3c",
          "url": "https://github.com/equinor/ert/commit/adf59e896075536f18cee1d6492cf56ded2539e6"
        },
        "date": 1705478402688,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.64586022395125,
            "unit": "iter/sec",
            "range": "stddev: 0.002170377045215055",
            "extra": "mean: 150.469610600004 msec\nrounds: 5"
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
          "id": "8a6fc5ff9bb0e2cf419d80fe8723add6ee4d19f4",
          "message": "Use read_summary for refcase\n\nEnsures consistent naming of summary keys. This also fixes an issue\nwhere history keys were not handled correctly: e.g.\n\nthe history key of \"BOPR:1,1,3\" was before interpreted to be\n\"BOPR:1,1,3H\" but is now interpreted as \"BOPRH:1,1,3\".\n\nWhether that always makes sense in all simulators is not confirmed, but\nit \"BOPR:1,1,3H\" is guaranteed to not be found in the summary file. The\nkeys that did work correctly before, FIELD, OTHER, GROUP and WELL still\nworks correctly.",
          "timestamp": "2024-01-17T10:36:28+01:00",
          "tree_id": "4eb334b4e9c27fe199f96201089049417cdb82ff",
          "url": "https://github.com/equinor/ert/commit/8a6fc5ff9bb0e2cf419d80fe8723add6ee4d19f4"
        },
        "date": 1705484336493,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.693489200349651,
            "unit": "iter/sec",
            "range": "stddev: 0.005149431104478457",
            "extra": "mean: 149.39891140001578 msec\nrounds: 5"
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
          "id": "91ea3633038762878bff47222162dc6b9aa6ceeb",
          "message": "Improve plotting performance",
          "timestamp": "2024-01-18T08:54:20+01:00",
          "tree_id": "ac0b76c5e7b9866bc195d3f3e6cd05e21a834fc2",
          "url": "https://github.com/equinor/ert/commit/91ea3633038762878bff47222162dc6b9aa6ceeb"
        },
        "date": 1705564607846,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.759337453527884,
            "unit": "iter/sec",
            "range": "stddev: 0.002240827460519864",
            "extra": "mean: 147.94349400000328 msec\nrounds: 5"
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
          "id": "ba54b66c91d407bb41ef4a0533bdb4f86fb90ed9",
          "message": "Provide more descriptive feedback on USE_EE and USE_GE\n\nCollect errors before raising them together",
          "timestamp": "2024-01-18T16:30:36+01:00",
          "tree_id": "c01091d041d2da574077b5e8f410aeda77419fa1",
          "url": "https://github.com/equinor/ert/commit/ba54b66c91d407bb41ef4a0533bdb4f86fb90ed9"
        },
        "date": 1705591992670,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.652320352422409,
            "unit": "iter/sec",
            "range": "stddev: 0.00295262134103806",
            "extra": "mean: 150.32348820000152 msec\nrounds: 5"
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
          "id": "f367789493dec64d5f530861c02026815086c894",
          "message": "Fix summary_keys generator",
          "timestamp": "2024-01-19T07:30:17+01:00",
          "tree_id": "bcb3b493489319c8115fb0c71bdd556fcb373d42",
          "url": "https://github.com/equinor/ert/commit/f367789493dec64d5f530861c02026815086c894"
        },
        "date": 1705645972408,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6747049693637255,
            "unit": "iter/sec",
            "range": "stddev: 0.0018697825168331522",
            "extra": "mean: 149.81935599998906 msec\nrounds: 5"
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
          "id": "c945790f77e743fd717fa64b3a28df72b6c6c68a",
          "message": "Refactor transfer functions test",
          "timestamp": "2024-01-19T09:23:04+01:00",
          "tree_id": "63b8ade52413ed037839a97a63e7d5666adc9c7b",
          "url": "https://github.com/equinor/ert/commit/c945790f77e743fd717fa64b3a28df72b6c6c68a"
        },
        "date": 1705652733558,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.348918354207445,
            "unit": "iter/sec",
            "range": "stddev: 0.010161119407034342",
            "extra": "mean: 157.50714440000593 msec\nrounds: 5"
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
          "id": "871a0b4ab277abb29da89c41fa794018bcca7a9a",
          "message": "Fix dark storage performance benchmark workflow trigger (#6971)",
          "timestamp": "2024-01-19T11:15:41Z",
          "tree_id": "77a58924323754bed877de768a6cfdb3a3eef055",
          "url": "https://github.com/equinor/ert/commit/871a0b4ab277abb29da89c41fa794018bcca7a9a"
        },
        "date": 1705663105027,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.680922597463627,
            "unit": "iter/sec",
            "range": "stddev: 0.001978600272888562",
            "extra": "mean: 149.67992599998752 msec\nrounds: 5"
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
          "id": "326c498d94089371e87472d98ae70f2d2e716564",
          "message": "Support submit sleep in scheduler (#6858)\n\nSupport submit_sleep in Scheduler\r\n\r\nNotes:\r\n* The test setup possibly allows flakyness\r\n* Picks SUBMIT_SLEEP from queue system configuration,\r\n  but leans towards a future global setting for SUBMIT_SLEEP.",
          "timestamp": "2024-01-19T12:36:12+01:00",
          "tree_id": "a7cca560836a83c1b7a754bcae4dc99857779050",
          "url": "https://github.com/equinor/ert/commit/326c498d94089371e87472d98ae70f2d2e716564"
        },
        "date": 1705664326276,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.639284901319427,
            "unit": "iter/sec",
            "range": "stddev: 0.0018467943508256963",
            "extra": "mean: 150.61863060000178 msec\nrounds: 5"
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
          "id": "bfb7547b9a8f729426ba25471df68005bbbcbf0d",
          "message": "Set minimum suggestor width to no more than two ui-cards",
          "timestamp": "2024-01-19T13:53:49+01:00",
          "tree_id": "dfb9a57ba92903490daa65bdc9457f0cd21dade9",
          "url": "https://github.com/equinor/ert/commit/bfb7547b9a8f729426ba25471df68005bbbcbf0d"
        },
        "date": 1705668978989,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.623896265630246,
            "unit": "iter/sec",
            "range": "stddev: 0.005638010839889703",
            "extra": "mean: 150.96854779999376 msec\nrounds: 5"
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
          "id": "c475601d3d1ca9160e7dd7eea58991ede009be7c",
          "message": "Rem. duplicit line",
          "timestamp": "2024-01-19T14:47:29+01:00",
          "tree_id": "43eb949c687dab2f1fd211d07e266440554fdf7e",
          "url": "https://github.com/equinor/ert/commit/c475601d3d1ca9160e7dd7eea58991ede009be7c"
        },
        "date": 1705672192577,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.714459986943197,
            "unit": "iter/sec",
            "range": "stddev: 0.002469054114971495",
            "extra": "mean: 148.9323046000095 msec\nrounds: 5"
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
          "id": "48b4ecd38e3fb6ec750078e2b0fa81c324464849",
          "message": "Remove redudant while loop",
          "timestamp": "2024-01-19T14:52:15+01:00",
          "tree_id": "a3f04fd648b72a6c7e3c0769ef83c2f580a1ad9e",
          "url": "https://github.com/equinor/ert/commit/48b4ecd38e3fb6ec750078e2b0fa81c324464849"
        },
        "date": 1705672479962,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7219642693819806,
            "unit": "iter/sec",
            "range": "stddev: 0.002273414660751701",
            "extra": "mean: 148.76603920001799 msec\nrounds: 5"
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
          "id": "8f2564638b74358e1a039175f17f365521de7db5",
          "message": "Make poly-case runnable by default on RHEL7 (#6964)\n\nWithout engaging a particular environment (like Komodo)",
          "timestamp": "2024-01-19T14:57:03+01:00",
          "tree_id": "4dac9f8ba5ec1b59a36165eb37c50987e64c3b0f",
          "url": "https://github.com/equinor/ert/commit/8f2564638b74358e1a039175f17f365521de7db5"
        },
        "date": 1705672785504,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.198054446575896,
            "unit": "iter/sec",
            "range": "stddev: 0.0033636438292381286",
            "extra": "mean: 161.34095120000893 msec\nrounds: 5"
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
          "id": "0465891188ae715f7cf6b56e71beaf28241d0ccf",
          "message": "Mark two scheduler tests as flaky (#6980)\n\nThese have been seen to be flaky but will pass on retries. Rerunning\r\ninstead of tuning the timing requirements will keep the timing\r\nrequirements easier to understand.",
          "timestamp": "2024-01-22T06:38:22Z",
          "tree_id": "62fac2b04e8e1117f863f8405d153dcb53db2e58",
          "url": "https://github.com/equinor/ert/commit/0465891188ae715f7cf6b56e71beaf28241d0ccf"
        },
        "date": 1705905686540,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.747599024781482,
            "unit": "iter/sec",
            "range": "stddev: 0.0021874596545612755",
            "extra": "mean: 148.2008631999861 msec\nrounds: 5"
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
          "id": "797b49af3c7814fffefffdf168aea733dc348494",
          "message": "Test triangular transfer function",
          "timestamp": "2024-01-23T08:46:52+01:00",
          "tree_id": "a1b72b99ac2ee1130fa71930bd97da46039d5d6f",
          "url": "https://github.com/equinor/ert/commit/797b49af3c7814fffefffdf168aea733dc348494"
        },
        "date": 1705996161446,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.616087927109271,
            "unit": "iter/sec",
            "range": "stddev: 0.0018765018683926482",
            "extra": "mean: 151.1467216000142 msec\nrounds: 5"
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
          "id": "747a6878219b38bba78ba8f5dd5d84cd6ce9e07c",
          "message": "Require version argument for ECLIPSE100 forward_model",
          "timestamp": "2024-01-23T12:01:35+01:00",
          "tree_id": "17d86621c1369c2a58314f737d7168bfa4b72bb6",
          "url": "https://github.com/equinor/ert/commit/747a6878219b38bba78ba8f5dd5d84cd6ce9e07c"
        },
        "date": 1706007843280,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.663378138518662,
            "unit": "iter/sec",
            "range": "stddev: 0.0022292487830246565",
            "extra": "mean: 150.07402840000168 msec\nrounds: 5"
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
          "id": "1df6dad8974c5f3b438850f3b0643d922ffdd948",
          "message": "Remove redudant test",
          "timestamp": "2024-01-23T12:02:16+01:00",
          "tree_id": "4310df2a4b47aff44b14f3af35c3e34b7f611a31",
          "url": "https://github.com/equinor/ert/commit/1df6dad8974c5f3b438850f3b0643d922ffdd948"
        },
        "date": 1706007887394,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.730702244748871,
            "unit": "iter/sec",
            "range": "stddev: 0.0025834030511434607",
            "extra": "mean: 148.57290720001401 msec\nrounds: 5"
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
          "id": "10cd8a9261298a2d042c86412beada964d4ec139",
          "message": "Add missing space in error message (#6989)",
          "timestamp": "2024-01-23T12:42:58+01:00",
          "tree_id": "e8d74754af816394e649c06d37063d69a483dcff",
          "url": "https://github.com/equinor/ert/commit/10cd8a9261298a2d042c86412beada964d4ec139"
        },
        "date": 1706010348007,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.627734997810887,
            "unit": "iter/sec",
            "range": "stddev: 0.0010747779787087122",
            "extra": "mean: 150.8811079999873 msec\nrounds: 5"
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
          "id": "aefd10a500b51c63f5cb93398dad96af4ff493b6",
          "message": "Add NaN for observations missing a response",
          "timestamp": "2024-01-23T15:08:30+01:00",
          "tree_id": "c955a8fd44f5ea9814c8067b37fca1c4d51a3ec7",
          "url": "https://github.com/equinor/ert/commit/aefd10a500b51c63f5cb93398dad96af4ff493b6"
        },
        "date": 1706019071248,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.652190223519401,
            "unit": "iter/sec",
            "range": "stddev: 0.0024028578549890414",
            "extra": "mean: 150.32642880000822 msec\nrounds: 5"
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
          "id": "433b916b70e5aa318eea3ae3e6a3eb62f723610e",
          "message": "Give proper error message when eclrun fails with nonzero return code (#6998)\n\nCatch return_code=1 properly from eclrun",
          "timestamp": "2024-01-23T21:34:00+01:00",
          "tree_id": "44a271586abe7de11ad319d41866c32e7010104f",
          "url": "https://github.com/equinor/ert/commit/433b916b70e5aa318eea3ae3e6a3eb62f723610e"
        },
        "date": 1706042202054,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.614623566918536,
            "unit": "iter/sec",
            "range": "stddev: 0.0029706218010351094",
            "extra": "mean: 151.18018280001024 msec\nrounds: 5"
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
          "id": "bb58b4685e1a1b31a95ebbaeefe4d4ffc8165a20",
          "message": "Remove unused endpoints",
          "timestamp": "2024-01-24T07:43:59+01:00",
          "tree_id": "d37c3ee1436681a7c32febdbe459707444666dce",
          "url": "https://github.com/equinor/ert/commit/bb58b4685e1a1b31a95ebbaeefe4d4ffc8165a20"
        },
        "date": 1706078799436,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.7024733346461405,
            "unit": "iter/sec",
            "range": "stddev: 0.0021070097234041608",
            "extra": "mean: 149.1986539999857 msec\nrounds: 5"
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
          "id": "1da097855b1222a5587b5f16d4312d907d1b290f",
          "message": "Relax requirement on return_code from mocked eclrun (#7011)\n\nThe caught return code is apparently sometimes translated from 1 to 255,\r\nboth variants observed on RHEL7.",
          "timestamp": "2024-01-24T11:36:49+01:00",
          "tree_id": "d8ad0e268625fccc0ccf41fcf2a5400ce917356a",
          "url": "https://github.com/equinor/ert/commit/1da097855b1222a5587b5f16d4312d907d1b290f"
        },
        "date": 1706092769719,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.68729051134624,
            "unit": "iter/sec",
            "range": "stddev: 0.0023085780859735984",
            "extra": "mean: 149.53739459999724 msec\nrounds: 5"
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
          "id": "789d7dc2a790f5b4ab7026250903bf00217b9f50",
          "message": "Match observations with 1 second tolerance\n\nSince summary files have precision loss of time, we need to include some\ntolerance when matching responses to observations.\n\nAlso contains a workaround for storage not handling datetimes with\nmicroseconds due to index overflow in netcdf3.\nhttps://github.com/equinor/ert/issues/6952",
          "timestamp": "2024-01-24T12:32:33+01:00",
          "tree_id": "ef031b81304a414be2750e0388f3fa24707f391d",
          "url": "https://github.com/equinor/ert/commit/789d7dc2a790f5b4ab7026250903bf00217b9f50"
        },
        "date": 1706096119649,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.6382896865103005,
            "unit": "iter/sec",
            "range": "stddev: 0.0015253361219466932",
            "extra": "mean: 150.6412113999943 msec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0795f09cef1d62cddfaf4ceb57cdd60a2ae4afc4",
          "message": "Scheduler: Pick last MAX_RUNNING rather than first (#7008)",
          "timestamp": "2024-01-24T12:41:57+01:00",
          "tree_id": "5fe8642980bf5da45c25ca81314442ac55d4d2e9",
          "url": "https://github.com/equinor/ert/commit/0795f09cef1d62cddfaf4ceb57cdd60a2ae4afc4"
        },
        "date": 1706096662616,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/unit_tests/analysis/test_es_update.py::test_and_benchmark_adaptive_localization_with_fields",
            "value": 6.680000053653901,
            "unit": "iter/sec",
            "range": "stddev: 0.002363945228645518",
            "extra": "mean: 149.70059759999685 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}