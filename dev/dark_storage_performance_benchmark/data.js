window.BENCHMARK_DATA = {
  "lastUpdate": 1706024350573,
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
      }
    ]
  }
}