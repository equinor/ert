window.BENCHMARK_DATA = {
  "lastUpdate": 1706100060959,
  "repoUrl": "https://github.com/equinor/ert",
  "entries": {
    "Python Dark Storage Benchmark": [
      {
        "commit": {
          "author": {
            "email": "f_scout_ci@st-linapp1196.st.statoil.no",
            "name": "Function Key"
          },
          "committer": {
            "email": "f_scout_ci@st-linapp1196.st.statoil.no",
            "name": "Function Key"
          },
          "distinct": true,
          "id": "3f415f330579a2354824d7b77652faf6d82e5420",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-23T13:56:25+01:00",
          "tree_id": "fd39ae24a98ba3e5c7f65337b0167c27f153cee7",
          "url": "https://github.com/equinor/ert/commit/3f415f330579a2354824d7b77652faf6d82e5420"
        },
        "date": 1706014621722,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8585255674361264,
            "unit": "iter/sec",
            "range": "stddev: 0.01916089274186424",
            "extra": "mean: 1.1647876754403115 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.841044641798613,
            "unit": "iter/sec",
            "range": "stddev: 0.02215844420198192",
            "extra": "mean: 1.1889975279569627 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8275195733903221,
            "unit": "iter/sec",
            "range": "stddev: 0.025724595284738784",
            "extra": "mean: 1.2084306307137012 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7195.89333638819,
            "unit": "iter/sec",
            "range": "stddev: 0.000012365107388385615",
            "extra": "mean: 138.96815214633608 usec\nrounds: 7883"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.9053259451921714,
            "unit": "iter/sec",
            "range": "stddev: 0.004619506636910899",
            "extra": "mean: 524.8445823788643 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.248000053382112,
            "unit": "iter/sec",
            "range": "stddev: 0.010814530330049636",
            "extra": "mean: 444.8398470878601 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.7790332476439794,
            "unit": "iter/sec",
            "range": "stddev: 0.06097822477828895",
            "extra": "mean: 562.1030418202281 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7312.979376361069,
            "unit": "iter/sec",
            "range": "stddev: 0.000012771048589309046",
            "extra": "mean: 136.74317245204634 usec\nrounds: 7969"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8377507261612854,
            "unit": "iter/sec",
            "range": "stddev: 0.022770671070092043",
            "extra": "mean: 1.1936724956147373 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8647826728471611,
            "unit": "iter/sec",
            "range": "stddev: 0.007769969107892465",
            "extra": "mean: 1.1563598941080273 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8880051482576283,
            "unit": "iter/sec",
            "range": "stddev: 0.00417476543153947",
            "extra": "mean: 1.1261195973493159 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 7264.993297151575,
            "unit": "iter/sec",
            "range": "stddev: 0.0000087334088340867",
            "extra": "mean: 137.64637613527813 usec\nrounds: 7808"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8817830597222975,
            "unit": "iter/sec",
            "range": "stddev: 0.0028732836348821867",
            "extra": "mean: 531.4108843915164 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.2665679925093065,
            "unit": "iter/sec",
            "range": "stddev: 0.0031317426370636747",
            "extra": "mean: 441.19567703455687 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.917271433439658,
            "unit": "iter/sec",
            "range": "stddev: 0.0063551914953787906",
            "extra": "mean: 521.5745577588677 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7082.645447223443,
            "unit": "iter/sec",
            "range": "stddev: 0.000015335603632391862",
            "extra": "mean: 141.19018203742257 usec\nrounds: 7907"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 728.8702669502619,
            "unit": "iter/sec",
            "range": "stddev: 0.000050497342523402546",
            "extra": "mean: 1.3719862715544686 msec\nrounds: 755"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1973.729040395339,
            "unit": "iter/sec",
            "range": "stddev: 0.00003273355828182867",
            "extra": "mean: 506.65515860257057 usec\nrounds: 2074"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.34905489614516,
            "unit": "iter/sec",
            "range": "stddev: 0.0009327394234812411",
            "extra": "mean: 74.91167036018201 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 382.5880486462284,
            "unit": "iter/sec",
            "range": "stddev: 0.0002498675297502108",
            "extra": "mean: 2.613777412907846 msec\nrounds: 402"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08207688269801919,
            "unit": "iter/sec",
            "range": "stddev: 0.17949687627843844",
            "extra": "mean: 12.18369859000668 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10443882233312994,
            "unit": "iter/sec",
            "range": "stddev: 0.07710504403287807",
            "extra": "mean: 9.574983494263142 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.07906445201869021,
            "unit": "iter/sec",
            "range": "stddev: 0.19754501859837317",
            "extra": "mean: 12.647909072507172 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7113.1333881985365,
            "unit": "iter/sec",
            "range": "stddev: 0.000012930289647506216",
            "extra": "mean: 140.58502005025082 usec\nrounds: 7852"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.20728825034705026,
            "unit": "iter/sec",
            "range": "stddev: 0.030351334064169937",
            "extra": "mean: 4.824200109392405 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.8049827959078827,
            "unit": "iter/sec",
            "range": "stddev: 0.027788354436714624",
            "extra": "mean: 1.2422625739127398 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.19811457710212502,
            "unit": "iter/sec",
            "range": "stddev: 0.3544448628497555",
            "extra": "mean: 5.047584153711796 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 6959.464259753008,
            "unit": "iter/sec",
            "range": "stddev: 0.000021273854364273967",
            "extra": "mean: 143.6892212785773 usec\nrounds: 7768"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.07781831433080641,
            "unit": "iter/sec",
            "range": "stddev: 0.44273584850865155",
            "extra": "mean: 12.850445407349616 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.09742341734914471,
            "unit": "iter/sec",
            "range": "stddev: 0.3159917091383039",
            "extra": "mean: 10.264472620747984 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08078225492640502,
            "unit": "iter/sec",
            "range": "stddev: 0.17880803880301996",
            "extra": "mean: 12.378956256061793 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7100.350393536878,
            "unit": "iter/sec",
            "range": "stddev: 0.00001476378515088985",
            "extra": "mean: 140.838119891978 usec\nrounds: 7893"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.2143629293248518,
            "unit": "iter/sec",
            "range": "stddev: 0.05406113835378555",
            "extra": "mean: 4.66498570041731 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.8140586137447345,
            "unit": "iter/sec",
            "range": "stddev: 0.009104215126226898",
            "extra": "mean: 1.2284127741120756 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.21317933563724148,
            "unit": "iter/sec",
            "range": "stddev: 0.03006198362500176",
            "extra": "mean: 4.690886182803661 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7020.986267538534,
            "unit": "iter/sec",
            "range": "stddev: 0.000021374738059375998",
            "extra": "mean: 142.43013187812244 usec\nrounds: 7785"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 672.1792421163553,
            "unit": "iter/sec",
            "range": "stddev: 0.0003226683382337905",
            "extra": "mean: 1.487698425276421 msec\nrounds: 724"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2020.0553811258383,
            "unit": "iter/sec",
            "range": "stddev: 0.0000389029184503915",
            "extra": "mean: 495.0359328478756 usec\nrounds: 2149"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.220518336669198,
            "unit": "iter/sec",
            "range": "stddev: 0.002368766617684811",
            "extra": "mean: 75.63999947160482 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 381.01666977366835,
            "unit": "iter/sec",
            "range": "stddev: 0.00007167727726321991",
            "extra": "mean: 2.6245570845864052 msec\nrounds: 393"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "committer": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "distinct": true,
          "id": "a5f695e10a512d085d400ebf11f9df87cdf4efda",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-23T16:38:52+01:00",
          "tree_id": "865bd087e99c129b55c9f68422584d822ab56f0e",
          "url": "https://github.com/equinor/ert/commit/a5f695e10a512d085d400ebf11f9df87cdf4efda"
        },
        "date": 1706024350531,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.869238911990367,
            "unit": "iter/sec",
            "range": "stddev: 0.006227958635576952",
            "extra": "mean: 1.1504317008890212 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8546682373266633,
            "unit": "iter/sec",
            "range": "stddev: 0.009679424673830761",
            "extra": "mean: 1.1700446516275407 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8605193046320389,
            "unit": "iter/sec",
            "range": "stddev: 0.0070271208381983",
            "extra": "mean: 1.1620889788493514 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7276.120175938633,
            "unit": "iter/sec",
            "range": "stddev: 0.000009097855233484352",
            "extra": "mean: 137.4358828358684 usec\nrounds: 7954"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.9198977527140226,
            "unit": "iter/sec",
            "range": "stddev: 0.00584411449243077",
            "extra": "mean: 520.8610711619258 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.309240545017445,
            "unit": "iter/sec",
            "range": "stddev: 0.00426596865508098",
            "extra": "mean: 433.04280368611217 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.8731147369839265,
            "unit": "iter/sec",
            "range": "stddev: 0.0100707948148343",
            "extra": "mean: 533.8701256550848 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 6979.146139074769,
            "unit": "iter/sec",
            "range": "stddev: 0.000023419290286883775",
            "extra": "mean: 143.28400352604893 usec\nrounds: 7785"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8612198970037102,
            "unit": "iter/sec",
            "range": "stddev: 0.01196217118195975",
            "extra": "mean: 1.161143632978201 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8609588524178339,
            "unit": "iter/sec",
            "range": "stddev: 0.005735708872379976",
            "extra": "mean: 1.1614956942386925 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8598036123866329,
            "unit": "iter/sec",
            "range": "stddev: 0.012951479178432031",
            "extra": "mean: 1.1630562905222177 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 6982.314266074864,
            "unit": "iter/sec",
            "range": "stddev: 0.00001497829031365388",
            "extra": "mean: 143.2189904225199 usec\nrounds: 7777"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8742655401091135,
            "unit": "iter/sec",
            "range": "stddev: 0.0034191147605190557",
            "extra": "mean: 533.542328234762 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.2465334474316414,
            "unit": "iter/sec",
            "range": "stddev: 0.008059134639438196",
            "extra": "mean: 445.13025218620896 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8391869646095613,
            "unit": "iter/sec",
            "range": "stddev: 0.022843345190492772",
            "extra": "mean: 543.7185121700168 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 6877.706518688925,
            "unit": "iter/sec",
            "range": "stddev: 0.000015802200017467167",
            "extra": "mean: 145.39730610526644 usec\nrounds: 7684"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 687.9275396136616,
            "unit": "iter/sec",
            "range": "stddev: 0.00023594133162240557",
            "extra": "mean: 1.4536414700908726 msec\nrounds: 735"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 2015.0040807471094,
            "unit": "iter/sec",
            "range": "stddev: 0.000036924575067393926",
            "extra": "mean: 496.2769105803632 usec\nrounds: 2135"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 12.646675695509805,
            "unit": "iter/sec",
            "range": "stddev: 0.004196070476818991",
            "extra": "mean: 79.07216284157974 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 393.26762066106113,
            "unit": "iter/sec",
            "range": "stddev: 0.000028787702164920087",
            "extra": "mean: 2.5427976966907555 msec\nrounds: 401"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08171187158540602,
            "unit": "iter/sec",
            "range": "stddev: 0.10641877312047986",
            "extra": "mean: 12.238123795203865 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10052138425945129,
            "unit": "iter/sec",
            "range": "stddev: 0.2084143527496383",
            "extra": "mean: 9.948132005613298 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.08160704908776611,
            "unit": "iter/sec",
            "range": "stddev: 0.08096806750248513",
            "extra": "mean: 12.253843401744962 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7103.397511497854,
            "unit": "iter/sec",
            "range": "stddev: 0.000012742460143943284",
            "extra": "mean: 140.77770508849582 usec\nrounds: 7825"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.21221099786213754,
            "unit": "iter/sec",
            "range": "stddev: 0.054683365546867486",
            "extra": "mean: 4.712291116267442 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7858057271664268,
            "unit": "iter/sec",
            "range": "stddev: 0.03533723298675434",
            "extra": "mean: 1.2725791699253022 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21343473686818112,
            "unit": "iter/sec",
            "range": "stddev: 0.0355226271262319",
            "extra": "mean: 4.685272953566164 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 6894.423760193452,
            "unit": "iter/sec",
            "range": "stddev: 0.000013352790396780276",
            "extra": "mean: 145.0447542510704 usec\nrounds: 7593"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08170426266151093,
            "unit": "iter/sec",
            "range": "stddev: 0.047997743771865456",
            "extra": "mean: 12.239263502601535 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10379226341670535,
            "unit": "iter/sec",
            "range": "stddev: 0.2540324292868511",
            "extra": "mean: 9.634629471227527 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08227024623476506,
            "unit": "iter/sec",
            "range": "stddev: 0.05967763445134113",
            "extra": "mean: 12.155062683857977 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7172.374806623967,
            "unit": "iter/sec",
            "range": "stddev: 0.000008480587000536064",
            "extra": "mean: 139.4238347773545 usec\nrounds: 7811"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.21052130415294096,
            "unit": "iter/sec",
            "range": "stddev: 0.04309938001011584",
            "extra": "mean: 4.7501130777411165 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.8062658387803071,
            "unit": "iter/sec",
            "range": "stddev: 0.0211658524663066",
            "extra": "mean: 1.240285712108016 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.20919107571073975,
            "unit": "iter/sec",
            "range": "stddev: 0.029238439603859866",
            "extra": "mean: 4.7803186469711365 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7175.435664462219,
            "unit": "iter/sec",
            "range": "stddev: 0.00000854182443150575",
            "extra": "mean: 139.36436012557397 usec\nrounds: 7708"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 671.6364306257284,
            "unit": "iter/sec",
            "range": "stddev: 0.0000660108912383005",
            "extra": "mean: 1.4889007719077307 msec\nrounds: 703"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 1785.0649844990912,
            "unit": "iter/sec",
            "range": "stddev: 0.000024579919541580544",
            "extra": "mean: 560.2036949263284 usec\nrounds: 1844"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.322058748472767,
            "unit": "iter/sec",
            "range": "stddev: 0.001593205856631986",
            "extra": "mean: 75.06347321239966 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 367.88611567540516,
            "unit": "iter/sec",
            "range": "stddev: 0.000033914467769511006",
            "extra": "mean: 2.7182325110696057 msec\nrounds: 374"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "committer": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "distinct": true,
          "id": "5b190da0e5613f0b6c3249432d598f8caaf55b78",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-23T22:38:47+01:00",
          "tree_id": "e1c39edfa6f2bb3aa144759525874d1c2752a6e8",
          "url": "https://github.com/equinor/ert/commit/5b190da0e5613f0b6c3249432d598f8caaf55b78"
        },
        "date": 1706045945370,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8620023515057301,
            "unit": "iter/sec",
            "range": "stddev: 0.006643644328314188",
            "extra": "mean: 1.160089642740786 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.843684160537734,
            "unit": "iter/sec",
            "range": "stddev: 0.01793515358293045",
            "extra": "mean: 1.1852776747196914 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8623010782980696,
            "unit": "iter/sec",
            "range": "stddev: 0.00700457741701684",
            "extra": "mean: 1.1596877531148493 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7158.663761262908,
            "unit": "iter/sec",
            "range": "stddev: 0.00000970575778549596",
            "extra": "mean: 139.69087435160986 usec\nrounds: 7723"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8924310916822085,
            "unit": "iter/sec",
            "range": "stddev: 0.012328564392063996",
            "extra": "mean: 528.4208256751299 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.254330426730777,
            "unit": "iter/sec",
            "range": "stddev: 0.006452425021930774",
            "extra": "mean: 443.5906946659088 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.9089789603948228,
            "unit": "iter/sec",
            "range": "stddev: 0.0012983560510761993",
            "extra": "mean: 523.8402416929603 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7194.993404441436,
            "unit": "iter/sec",
            "range": "stddev: 0.000008914318189277442",
            "extra": "mean: 138.98553393846123 usec\nrounds: 7687"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8567346738459237,
            "unit": "iter/sec",
            "range": "stddev: 0.05039396486375183",
            "extra": "mean: 1.1672225141897798 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8506971286214686,
            "unit": "iter/sec",
            "range": "stddev: 0.017255507469574723",
            "extra": "mean: 1.1755064950324594 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8682229723011923,
            "unit": "iter/sec",
            "range": "stddev: 0.007442132043815301",
            "extra": "mean: 1.1517778634093703 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 6758.717061939978,
            "unit": "iter/sec",
            "range": "stddev: 0.000026017130021315514",
            "extra": "mean: 147.95707392920022 usec\nrounds: 7665"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8860769324903344,
            "unit": "iter/sec",
            "range": "stddev: 0.00644679322062428",
            "extra": "mean: 530.2010659128428 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.2138838224394672,
            "unit": "iter/sec",
            "range": "stddev: 0.009236198759194065",
            "extra": "mean: 451.69488564133644 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.795850899916126,
            "unit": "iter/sec",
            "range": "stddev: 0.03960486585430521",
            "extra": "mean: 556.8391006439924 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 6978.513990113221,
            "unit": "iter/sec",
            "range": "stddev: 0.000015809332748112434",
            "extra": "mean: 143.29698291308804 usec\nrounds: 7681"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 727.0921681649669,
            "unit": "iter/sec",
            "range": "stddev: 0.00003829971180049859",
            "extra": "mean: 1.3753414543355584 msec\nrounds: 744"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1922.5901365589075,
            "unit": "iter/sec",
            "range": "stddev: 0.000024063811358836755",
            "extra": "mean: 520.1316604015358 usec\nrounds: 2089"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.310392770279716,
            "unit": "iter/sec",
            "range": "stddev: 0.0010318620993416796",
            "extra": "mean: 75.12926306974676 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 381.07593213288106,
            "unit": "iter/sec",
            "range": "stddev: 0.00007887675874887473",
            "extra": "mean: 2.624148931167084 msec\nrounds: 398"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08092914088316179,
            "unit": "iter/sec",
            "range": "stddev: 0.23772082864843969",
            "extra": "mean: 12.356488516833632 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.1034398611304326,
            "unit": "iter/sec",
            "range": "stddev: 0.06473238375171127",
            "extra": "mean: 9.66745304055512 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.08165027006896687,
            "unit": "iter/sec",
            "range": "stddev: 0.02619644581340699",
            "extra": "mean: 12.247356918174773 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7156.0971304068225,
            "unit": "iter/sec",
            "range": "stddev: 0.00000842105875528421",
            "extra": "mean: 139.74097636977575 usec\nrounds: 7639"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.21384795782388893,
            "unit": "iter/sec",
            "range": "stddev: 0.03615365160897443",
            "extra": "mean: 4.676219544839114 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.8041138583229575,
            "unit": "iter/sec",
            "range": "stddev: 0.021259523919521163",
            "extra": "mean: 1.2436049816198647 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21733225509959792,
            "unit": "iter/sec",
            "range": "stddev: 0.040871460068598785",
            "extra": "mean: 4.6012498215772215 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 7027.203650396626,
            "unit": "iter/sec",
            "range": "stddev: 0.000012258310821786092",
            "extra": "mean: 142.30411551308302 usec\nrounds: 7585"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08070801897747293,
            "unit": "iter/sec",
            "range": "stddev: 0.12892857236591765",
            "extra": "mean: 12.390342529397458 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10370046187379248,
            "unit": "iter/sec",
            "range": "stddev: 0.03715018000973651",
            "extra": "mean: 9.64315859284252 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08208512492781478,
            "unit": "iter/sec",
            "range": "stddev: 0.10125260889277245",
            "extra": "mean: 12.182475215569138 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 6646.342256509232,
            "unit": "iter/sec",
            "range": "stddev: 0.000024965192757483907",
            "extra": "mean: 150.4586976423956 usec\nrounds: 7563"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.21453907679452144,
            "unit": "iter/sec",
            "range": "stddev: 0.01783549532865504",
            "extra": "mean: 4.661155510414392 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.7962545077985937,
            "unit": "iter/sec",
            "range": "stddev: 0.027777513956718066",
            "extra": "mean: 1.255879860278219 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.21007422564849146,
            "unit": "iter/sec",
            "range": "stddev: 0.037334089329139154",
            "extra": "mean: 4.760222235321999 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7048.166569429448,
            "unit": "iter/sec",
            "range": "stddev: 0.000007277738915690911",
            "extra": "mean: 141.88086932243863 usec\nrounds: 7691"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 724.8488452444825,
            "unit": "iter/sec",
            "range": "stddev: 0.00007194462102661871",
            "extra": "mean: 1.3795979762687107 msec\nrounds: 738"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2119.7220952051757,
            "unit": "iter/sec",
            "range": "stddev: 0.000014093590092652746",
            "extra": "mean: 471.7599548837115 usec\nrounds: 2179"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 12.70531945048969,
            "unit": "iter/sec",
            "range": "stddev: 0.007066717117190393",
            "extra": "mean: 78.70719062962702 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 419.30559258629233,
            "unit": "iter/sec",
            "range": "stddev: 0.00010608646142994504",
            "extra": "mean: 2.384895450194125 msec\nrounds: 432"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "committer": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "distinct": true,
          "id": "4242c0ed0edf5c8a937530b159c94bfa76a03764",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-24T10:39:17+01:00",
          "tree_id": "47745dedc98a3fa810357f27c8e7c5f43041edd1",
          "url": "https://github.com/equinor/ert/commit/4242c0ed0edf5c8a937530b159c94bfa76a03764"
        },
        "date": 1706089176093,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8454477398141591,
            "unit": "iter/sec",
            "range": "stddev: 0.006281026564357985",
            "extra": "mean: 1.182805220130831 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.819849776671751,
            "unit": "iter/sec",
            "range": "stddev: 0.03949132296177643",
            "extra": "mean: 1.2197356496937573 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8405823165902769,
            "unit": "iter/sec",
            "range": "stddev: 0.017618807406407217",
            "extra": "mean: 1.1896514835767449 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 6713.460498466363,
            "unit": "iter/sec",
            "range": "stddev: 0.000016031277570477365",
            "extra": "mean: 148.95447738590883 usec\nrounds: 7741"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8682347726317932,
            "unit": "iter/sec",
            "range": "stddev: 0.004718212711513464",
            "extra": "mean: 535.2646330371499 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.247402509111486,
            "unit": "iter/sec",
            "range": "stddev: 0.0037558306275740925",
            "extra": "mean: 444.95812207460403 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.832030869640756,
            "unit": "iter/sec",
            "range": "stddev: 0.021975350668847222",
            "extra": "mean: 545.8423308096826 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 6824.493812130229,
            "unit": "iter/sec",
            "range": "stddev: 0.000013397676253893603",
            "extra": "mean: 146.53101424497524 usec\nrounds: 7654"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8376691762002267,
            "unit": "iter/sec",
            "range": "stddev: 0.021077772698657817",
            "extra": "mean: 1.1937887037172914 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8339152052211348,
            "unit": "iter/sec",
            "range": "stddev: 0.009938846646059958",
            "extra": "mean: 1.1991626891307532 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8492010132746968,
            "unit": "iter/sec",
            "range": "stddev: 0.012881454467087692",
            "extra": "mean: 1.1775774926878513 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 6635.346789667629,
            "unit": "iter/sec",
            "range": "stddev: 0.00003057785395255024",
            "extra": "mean: 150.70802351388343 usec\nrounds: 7631"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8368340665237004,
            "unit": "iter/sec",
            "range": "stddev: 0.00756316539003256",
            "extra": "mean: 544.4149900227785 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.21051829611268,
            "unit": "iter/sec",
            "range": "stddev: 0.008880014098808092",
            "extra": "mean: 452.38259360194206 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8358152199638997,
            "unit": "iter/sec",
            "range": "stddev: 0.009681296998059157",
            "extra": "mean: 544.7171311825514 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 6575.833568802082,
            "unit": "iter/sec",
            "range": "stddev: 0.000020494776801514345",
            "extra": "mean: 152.07197529212556 usec\nrounds: 7725"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 723.4367861848405,
            "unit": "iter/sec",
            "range": "stddev: 0.00006882144281090189",
            "extra": "mean: 1.3822907807517777 msec\nrounds: 772"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1914.405799666031,
            "unit": "iter/sec",
            "range": "stddev: 0.00004500002757193681",
            "extra": "mean: 522.355291743501 usec\nrounds: 2170"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 12.588045532464248,
            "unit": "iter/sec",
            "range": "stddev: 0.0012498060152631876",
            "extra": "mean: 79.44044986340616 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 425.22797713609964,
            "unit": "iter/sec",
            "range": "stddev: 0.00007632651367977091",
            "extra": "mean: 2.3516796959950197 msec\nrounds: 439"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08035851229014802,
            "unit": "iter/sec",
            "range": "stddev: 0.13121628387492",
            "extra": "mean: 12.444232371915131 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10132531648544539,
            "unit": "iter/sec",
            "range": "stddev: 0.06654239957210058",
            "extra": "mean: 9.869201841019095 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.07977041110967575,
            "unit": "iter/sec",
            "range": "stddev: 0.0970037576180809",
            "extra": "mean: 12.535976511705666 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 6740.9204100298875,
            "unit": "iter/sec",
            "range": "stddev: 0.000016915294432951995",
            "extra": "mean: 148.34769425731378 usec\nrounds: 7826"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.2078581781117352,
            "unit": "iter/sec",
            "range": "stddev: 0.05144403826959227",
            "extra": "mean: 4.810972602013498 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7806407294222231,
            "unit": "iter/sec",
            "range": "stddev: 0.01932325832565078",
            "extra": "mean: 1.2809990080073477 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21185618954197422,
            "unit": "iter/sec",
            "range": "stddev: 0.0334694541124504",
            "extra": "mean: 4.720183074008673 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 6866.277001539992,
            "unit": "iter/sec",
            "range": "stddev: 0.00001726029854988966",
            "extra": "mean: 145.63933260713432 usec\nrounds: 7622"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08059256380446302,
            "unit": "iter/sec",
            "range": "stddev: 0.16795793801113007",
            "extra": "mean: 12.408092667534948 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10044661217835045,
            "unit": "iter/sec",
            "range": "stddev: 0.19196439508880797",
            "extra": "mean: 9.955537357740104 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08083460797623053,
            "unit": "iter/sec",
            "range": "stddev: 0.09408442108103547",
            "extra": "mean: 12.370938946027309 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 6852.30490971736,
            "unit": "iter/sec",
            "range": "stddev: 0.000018665329847817752",
            "extra": "mean: 145.93629635217843 usec\nrounds: 7729"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.2115148495334409,
            "unit": "iter/sec",
            "range": "stddev: 0.034452532626855034",
            "extra": "mean: 4.72780044618994 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.8166317872950014,
            "unit": "iter/sec",
            "range": "stddev: 0.012952377274875871",
            "extra": "mean: 1.2245420954190194 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.21326480792902414,
            "unit": "iter/sec",
            "range": "stddev: 0.04927613758283017",
            "extra": "mean: 4.689006168954075 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 6978.034261099595,
            "unit": "iter/sec",
            "range": "stddev: 0.000012724524228281409",
            "extra": "mean: 143.3068343580217 usec\nrounds: 7719"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 756.3253583272667,
            "unit": "iter/sec",
            "range": "stddev: 0.00002904822473170285",
            "extra": "mean: 1.322182297591685 msec\nrounds: 778"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2162.994717744307,
            "unit": "iter/sec",
            "range": "stddev: 0.000008187478302539727",
            "extra": "mean: 462.3219797054597 usec\nrounds: 2231"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 12.973423840090806,
            "unit": "iter/sec",
            "range": "stddev: 0.0003426239181614614",
            "extra": "mean: 77.08065444603562 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 399.78015529799933,
            "unit": "iter/sec",
            "range": "stddev: 0.00003289501676008126",
            "extra": "mean: 2.5013747849854933 msec\nrounds: 406"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "committer": {
            "email": "f_scout_ci@st-linapp1192.st.statoil.no",
            "name": "Function Key"
          },
          "distinct": true,
          "id": "1ae3d595db949da183142a4a8982871f22199303",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-24T13:40:42+01:00",
          "tree_id": "9e01368ffaf037ce3b9c51daaeb8c7138f5406c1",
          "url": "https://github.com/equinor/ert/commit/1ae3d595db949da183142a4a8982871f22199303"
        },
        "date": 1706100060916,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8568895110316508,
            "unit": "iter/sec",
            "range": "stddev: 0.008487686869621423",
            "extra": "mean: 1.1670116008259357 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8524164854847618,
            "unit": "iter/sec",
            "range": "stddev: 0.02240999682729993",
            "extra": "mean: 1.173135453183204 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8707912269396427,
            "unit": "iter/sec",
            "range": "stddev: 0.004802159072626983",
            "extra": "mean: 1.1483808851800859 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7452.622322397323,
            "unit": "iter/sec",
            "range": "stddev: 0.000007699221995083585",
            "extra": "mean: 134.1809576200723 usec\nrounds: 7976"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8994830623315482,
            "unit": "iter/sec",
            "range": "stddev: 0.003553158911063325",
            "extra": "mean: 526.4590244740248 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.288614790259069,
            "unit": "iter/sec",
            "range": "stddev: 0.006282619049506219",
            "extra": "mean: 436.945528909564 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.8666906397639893,
            "unit": "iter/sec",
            "range": "stddev: 0.0054071283713070266",
            "extra": "mean: 535.7074057683349 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7092.388715782804,
            "unit": "iter/sec",
            "range": "stddev: 0.000014568789362968655",
            "extra": "mean: 140.99622004285865 usec\nrounds: 7886"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8705157302020899,
            "unit": "iter/sec",
            "range": "stddev: 0.00930063079273829",
            "extra": "mean: 1.1487443193793296 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8581880617873006,
            "unit": "iter/sec",
            "range": "stddev: 0.013201543613399513",
            "extra": "mean: 1.1652457596734167 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8654785488188409,
            "unit": "iter/sec",
            "range": "stddev: 0.010534688117828485",
            "extra": "mean: 1.155430139042437 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 7236.611357603214,
            "unit": "iter/sec",
            "range": "stddev: 0.000009131843964029287",
            "extra": "mean: 138.18622426770793 usec\nrounds: 7857"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8695261539764998,
            "unit": "iter/sec",
            "range": "stddev: 0.007874973379858348",
            "extra": "mean: 534.8948972299695 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.276983861804307,
            "unit": "iter/sec",
            "range": "stddev: 0.0038915582623729396",
            "extra": "mean: 439.17746488004923 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8940891584584278,
            "unit": "iter/sec",
            "range": "stddev: 0.003762189007673316",
            "extra": "mean: 527.958251349628 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7104.353977099079,
            "unit": "iter/sec",
            "range": "stddev: 0.000017762902226530977",
            "extra": "mean: 140.7587520587382 usec\nrounds: 7849"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 740.901697511845,
            "unit": "iter/sec",
            "range": "stddev: 0.00004007503360235823",
            "extra": "mean: 1.3497067200119524 msec\nrounds: 767"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1946.205817535819,
            "unit": "iter/sec",
            "range": "stddev: 0.00005113789593842574",
            "extra": "mean: 513.8202706978577 usec\nrounds: 2134"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 11.922383296824568,
            "unit": "iter/sec",
            "range": "stddev: 0.0009523485161961421",
            "extra": "mean: 83.87584722815798 msec\nrounds: 13"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 379.21122316961004,
            "unit": "iter/sec",
            "range": "stddev: 0.00012412812515083817",
            "extra": "mean: 2.6370527529263796 msec\nrounds: 395"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08081132882254,
            "unit": "iter/sec",
            "range": "stddev: 0.22230032457433094",
            "extra": "mean: 12.374502617027611 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10001786341305319,
            "unit": "iter/sec",
            "range": "stddev: 0.3235929915225261",
            "extra": "mean: 9.998213977739216 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.0767654092224136,
            "unit": "iter/sec",
            "range": "stddev: 0.11884638130229425",
            "extra": "mean: 13.026700569037349 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7146.884882209738,
            "unit": "iter/sec",
            "range": "stddev: 0.000022520402234203787",
            "extra": "mean: 139.92110079864767 usec\nrounds: 7921"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.2041676143820785,
            "unit": "iter/sec",
            "range": "stddev: 0.04478190886852829",
            "extra": "mean: 4.897936448082328 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7937843593349788,
            "unit": "iter/sec",
            "range": "stddev: 0.015772386106444585",
            "extra": "mean: 1.2597879868000745 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.2116803749970943,
            "unit": "iter/sec",
            "range": "stddev: 0.01718642536279269",
            "extra": "mean: 4.7241034980863335 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 7179.50784790397,
            "unit": "iter/sec",
            "range": "stddev: 0.000009000819779515986",
            "extra": "mean: 139.28531330903778 usec\nrounds: 7835"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08066131760476336,
            "unit": "iter/sec",
            "range": "stddev: 0.1373666995511339",
            "extra": "mean: 12.397516302671283 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10486216344979474,
            "unit": "iter/sec",
            "range": "stddev: 0.06676635254466412",
            "extra": "mean: 9.536328138783574 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08198333293970052,
            "unit": "iter/sec",
            "range": "stddev: 0.07576842088413846",
            "extra": "mean: 12.197601197008044 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7331.312903796522,
            "unit": "iter/sec",
            "range": "stddev: 0.000009576592092044337",
            "extra": "mean: 136.40121668823463 usec\nrounds: 7866"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.21157905180338135,
            "unit": "iter/sec",
            "range": "stddev: 0.045437109327526364",
            "extra": "mean: 4.726365826278925 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.8017123119487696,
            "unit": "iter/sec",
            "range": "stddev: 0.02947305982199144",
            "extra": "mean: 1.2473302269354463 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.20918007148010462,
            "unit": "iter/sec",
            "range": "stddev: 0.046805903383974945",
            "extra": "mean: 4.780570122785866 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7234.532035585881,
            "unit": "iter/sec",
            "range": "stddev: 0.00001156900229880002",
            "extra": "mean: 138.2259412331175 usec\nrounds: 7810"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 724.2568800850029,
            "unit": "iter/sec",
            "range": "stddev: 0.000019882218187238205",
            "extra": "mean: 1.3807255788617907 msec\nrounds: 738"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2003.2860636051407,
            "unit": "iter/sec",
            "range": "stddev: 0.000012027352223860033",
            "extra": "mean: 499.1798316613786 usec\nrounds: 2046"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.80432848229237,
            "unit": "iter/sec",
            "range": "stddev: 0.0015941209933524004",
            "extra": "mean: 72.441046392279 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 384.45330814003256,
            "unit": "iter/sec",
            "range": "stddev: 0.000060927836168141896",
            "extra": "mean: 2.601096098868167 msec\nrounds: 398"
          }
        ]
      }
    ]
  }
}