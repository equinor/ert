window.BENCHMARK_DATA = {
  "lastUpdate": 1706272821843,
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
          "id": "04c56af9b670c21c9eb1b70871e698fae1790644",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-24T16:39:14+01:00",
          "tree_id": "8714ba48b196216ec84e91522838c7d04a07a717",
          "url": "https://github.com/equinor/ert/commit/04c56af9b670c21c9eb1b70871e698fae1790644"
        },
        "date": 1706110779736,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8663073959899957,
            "unit": "iter/sec",
            "range": "stddev: 0.005189807718913414",
            "extra": "mean: 1.1543246711604298 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8443820821707776,
            "unit": "iter/sec",
            "range": "stddev: 0.009355986310239029",
            "extra": "mean: 1.1842979867942631 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8472953517755902,
            "unit": "iter/sec",
            "range": "stddev: 0.023390845829560393",
            "extra": "mean: 1.1802259954623877 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7104.587147208386,
            "unit": "iter/sec",
            "range": "stddev: 0.000011658583891374345",
            "extra": "mean: 140.75413240485497 usec\nrounds: 7834"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.9417708485566276,
            "unit": "iter/sec",
            "range": "stddev: 0.0008995691834750057",
            "extra": "mean: 514.9938267655671 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.3226231293939468,
            "unit": "iter/sec",
            "range": "stddev: 0.004850301485107246",
            "extra": "mean: 430.54768005385995 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.9033923580864527,
            "unit": "iter/sec",
            "range": "stddev: 0.010057616297482135",
            "extra": "mean: 525.3777529112995 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7123.751755901126,
            "unit": "iter/sec",
            "range": "stddev: 0.000012837204827629146",
            "extra": "mean: 140.37546987395044 usec\nrounds: 7844"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8558942031596026,
            "unit": "iter/sec",
            "range": "stddev: 0.00893308384107285",
            "extra": "mean: 1.1683687029406429 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8527137721660544,
            "unit": "iter/sec",
            "range": "stddev: 0.01809810006617587",
            "extra": "mean: 1.1727264559827746 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8625054951878546,
            "unit": "iter/sec",
            "range": "stddev: 0.007894484015751265",
            "extra": "mean: 1.1594129029661417 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 7114.445458427502,
            "unit": "iter/sec",
            "range": "stddev: 0.000010684248777762528",
            "extra": "mean: 140.559092882698 usec\nrounds: 7620"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8727354134299785,
            "unit": "iter/sec",
            "range": "stddev: 0.004239282141832482",
            "extra": "mean: 533.9782613329589 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.238978165714651,
            "unit": "iter/sec",
            "range": "stddev: 0.004282001334636193",
            "extra": "mean: 446.6323143802583 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8947277056984009,
            "unit": "iter/sec",
            "range": "stddev: 0.0037588493644953487",
            "extra": "mean: 527.7803227305412 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 6704.886849083436,
            "unit": "iter/sec",
            "range": "stddev: 0.000022717045206414542",
            "extra": "mean: 149.14494793252192 usec\nrounds: 7641"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 748.0677948026132,
            "unit": "iter/sec",
            "range": "stddev: 0.000010948070007514631",
            "extra": "mean: 1.3367772372340427 msec\nrounds: 752"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1921.5311945562964,
            "unit": "iter/sec",
            "range": "stddev: 0.000024131175149314706",
            "extra": "mean: 520.4183012136378 usec\nrounds: 2019"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.65211570879766,
            "unit": "iter/sec",
            "range": "stddev: 0.0007859948191936402",
            "extra": "mean: 73.24871992958444 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 379.4095663894456,
            "unit": "iter/sec",
            "range": "stddev: 0.000021704728577713064",
            "extra": "mean: 2.6356741858573707 msec\nrounds: 385"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08324525321383591,
            "unit": "iter/sec",
            "range": "stddev: 0.052108568138906505",
            "extra": "mean: 12.012696957401932 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.1043247124516736,
            "unit": "iter/sec",
            "range": "stddev: 0.09146936868029197",
            "extra": "mean: 9.58545656632632 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.08162546737080377,
            "unit": "iter/sec",
            "range": "stddev: 0.10288341262245145",
            "extra": "mean: 12.251078397594393 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7170.761940235582,
            "unit": "iter/sec",
            "range": "stddev: 0.000009895831226284465",
            "extra": "mean: 139.45519434816808 usec\nrounds: 7864"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.2133591522010484,
            "unit": "iter/sec",
            "range": "stddev: 0.01914989318698947",
            "extra": "mean: 4.686932759545743 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.8052936511851749,
            "unit": "iter/sec",
            "range": "stddev: 0.017310851067127153",
            "extra": "mean: 1.2417830421589315 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21320711897449887,
            "unit": "iter/sec",
            "range": "stddev: 0.04228429374057755",
            "extra": "mean: 4.690274906437844 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 7003.204834383524,
            "unit": "iter/sec",
            "range": "stddev: 0.000014351632395610076",
            "extra": "mean: 142.79176800460206 usec\nrounds: 7838"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.0809923427191933,
            "unit": "iter/sec",
            "range": "stddev: 0.15395227482698132",
            "extra": "mean: 12.346846213191748 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10484760364721366,
            "unit": "iter/sec",
            "range": "stddev: 0.09914593443192421",
            "extra": "mean: 9.537652413733303 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08122115502587803,
            "unit": "iter/sec",
            "range": "stddev: 0.0748449584077327",
            "extra": "mean: 12.312063275650143 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7111.0508783942705,
            "unit": "iter/sec",
            "range": "stddev: 0.000009560417956022361",
            "extra": "mean: 140.6261911355931 usec\nrounds: 7779"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.20899945374353995,
            "unit": "iter/sec",
            "range": "stddev: 0.02321149087683497",
            "extra": "mean: 4.784701500833035 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.7679855294291522,
            "unit": "iter/sec",
            "range": "stddev: 0.013318308283195747",
            "extra": "mean: 1.3021078675054014 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.2084686881971193,
            "unit": "iter/sec",
            "range": "stddev: 0.03452226286003638",
            "extra": "mean: 4.796883448772133 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7066.207803780995,
            "unit": "iter/sec",
            "range": "stddev: 0.00001409753530621619",
            "extra": "mean: 141.51862325148699 usec\nrounds: 7824"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 719.0976502008632,
            "unit": "iter/sec",
            "range": "stddev: 0.00008020809524980853",
            "extra": "mean: 1.390631717014613 msec\nrounds: 754"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2045.1933292281747,
            "unit": "iter/sec",
            "range": "stddev: 0.000009871290347949265",
            "extra": "mean: 488.9513307660675 usec\nrounds: 2078"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.373220933339297,
            "unit": "iter/sec",
            "range": "stddev: 0.0011964441895758594",
            "extra": "mean: 74.77630145980844 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 416.88627592972193,
            "unit": "iter/sec",
            "range": "stddev: 0.00003734912628518781",
            "extra": "mean: 2.39873571700062 msec\nrounds: 423"
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
          "id": "d96f3fed03ad82fafca42182bce2af481aca1c17",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-25T10:38:53+01:00",
          "tree_id": "991e454fa28e62f70e0176a72fa83c346102639f",
          "url": "https://github.com/equinor/ert/commit/d96f3fed03ad82fafca42182bce2af481aca1c17"
        },
        "date": 1706175553547,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8764581679658787,
            "unit": "iter/sec",
            "range": "stddev: 0.011768062024568128",
            "extra": "mean: 1.1409557655453682 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8494917085708492,
            "unit": "iter/sec",
            "range": "stddev: 0.017150612296225283",
            "extra": "mean: 1.177174526732415 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8498687420110651,
            "unit": "iter/sec",
            "range": "stddev: 0.0051453666063953655",
            "extra": "mean: 1.1766522882506252 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7321.536667713557,
            "unit": "iter/sec",
            "range": "stddev: 0.000010824939676557726",
            "extra": "mean: 136.5833492864675 usec\nrounds: 8005"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8495305324801743,
            "unit": "iter/sec",
            "range": "stddev: 0.009828281639779782",
            "extra": "mean: 540.6777462922037 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.2771761547795126,
            "unit": "iter/sec",
            "range": "stddev: 0.0050670325418710185",
            "extra": "mean: 439.14037914946675 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.8672834355948047,
            "unit": "iter/sec",
            "range": "stddev: 0.0027955115450009143",
            "extra": "mean: 535.5373377911747 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7448.593809541667,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071991240864675145",
            "extra": "mean: 134.25352832624563 usec\nrounds: 7980"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8139022404467234,
            "unit": "iter/sec",
            "range": "stddev: 0.008293256564805106",
            "extra": "mean: 1.228648786433041 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.812853894999067,
            "unit": "iter/sec",
            "range": "stddev: 0.027536389920409084",
            "extra": "mean: 1.2302333865314723 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8098220939516739,
            "unit": "iter/sec",
            "range": "stddev: 0.0037341109408809132",
            "extra": "mean: 1.2348391177132725 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 7321.491971761951,
            "unit": "iter/sec",
            "range": "stddev: 0.000011390682728702455",
            "extra": "mean: 136.58418309504003 usec\nrounds: 7896"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8039457162806878,
            "unit": "iter/sec",
            "range": "stddev: 0.0033604510223527043",
            "extra": "mean: 554.3404055759311 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.262960918884749,
            "unit": "iter/sec",
            "range": "stddev: 0.005063804683231103",
            "extra": "mean: 441.8989261612296 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8061850829019386,
            "unit": "iter/sec",
            "range": "stddev: 0.021199569658031853",
            "extra": "mean: 553.6531164310873 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7220.67459715126,
            "unit": "iter/sec",
            "range": "stddev: 0.000017536883817619188",
            "extra": "mean: 138.4912152660259 usec\nrounds: 7980"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 755.6149625396052,
            "unit": "iter/sec",
            "range": "stddev: 0.000008871984674222997",
            "extra": "mean: 1.323425354944034 msec\nrounds: 770"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 2120.9409652462136,
            "unit": "iter/sec",
            "range": "stddev: 0.00002641138887812696",
            "extra": "mean: 471.48884216299393 usec\nrounds: 2241"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.536374928777933,
            "unit": "iter/sec",
            "range": "stddev: 0.0017761506895493456",
            "extra": "mean: 73.87502232034292 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 399.4174945147471,
            "unit": "iter/sec",
            "range": "stddev: 0.00002231115715663346",
            "extra": "mean: 2.5036459687748565 msec\nrounds: 406"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08329757886344633,
            "unit": "iter/sec",
            "range": "stddev: 0.05275937261368187",
            "extra": "mean: 12.005150853656232 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10457591059150143,
            "unit": "iter/sec",
            "range": "stddev: 0.07183328042972555",
            "extra": "mean: 9.562431676127016 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.08228660467460445,
            "unit": "iter/sec",
            "range": "stddev: 0.0305905593209236",
            "extra": "mean: 12.152646277658642 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7151.754046351497,
            "unit": "iter/sec",
            "range": "stddev: 0.00001616712361874716",
            "extra": "mean: 139.825837622332 usec\nrounds: 7963"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.21455702746446556,
            "unit": "iter/sec",
            "range": "stddev: 0.04323754767309751",
            "extra": "mean: 4.660765540134162 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7850585347460173,
            "unit": "iter/sec",
            "range": "stddev: 0.020652298825518646",
            "extra": "mean: 1.2737903681583702 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21389645909071714,
            "unit": "iter/sec",
            "range": "stddev: 0.03889904704147279",
            "extra": "mean: 4.675159206707031 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 7392.30753049838,
            "unit": "iter/sec",
            "range": "stddev: 0.000008489889490079592",
            "extra": "mean: 135.27575738351098 usec\nrounds: 8005"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08011696050796115,
            "unit": "iter/sec",
            "range": "stddev: 0.17680227257984008",
            "extra": "mean: 12.481751599907875 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10348177742501559,
            "unit": "iter/sec",
            "range": "stddev: 0.030922381883373026",
            "extra": "mean: 9.663537145219744 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.07955667373321919,
            "unit": "iter/sec",
            "range": "stddev: 0.21363625847710055",
            "extra": "mean: 12.569655731879175 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7341.157983483164,
            "unit": "iter/sec",
            "range": "stddev: 0.000010947228379070323",
            "extra": "mean: 136.21829175313965 usec\nrounds: 7930"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.21177185251909197,
            "unit": "iter/sec",
            "range": "stddev: 0.01007032114568024",
            "extra": "mean: 4.722062862012535 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.7981791866287449,
            "unit": "iter/sec",
            "range": "stddev: 0.005788039324861971",
            "extra": "mean: 1.252851510979235 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.20958300787968215,
            "unit": "iter/sec",
            "range": "stddev: 0.027437458983964696",
            "extra": "mean: 4.771379178669304 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7288.601962688596,
            "unit": "iter/sec",
            "range": "stddev: 0.000013547080933525267",
            "extra": "mean: 137.20052283265628 usec\nrounds: 8002"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 742.2077822305641,
            "unit": "iter/sec",
            "range": "stddev: 0.00004001358394938872",
            "extra": "mean: 1.3473316016637422 msec\nrounds: 765"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 1978.634152765329,
            "unit": "iter/sec",
            "range": "stddev: 0.00007418717081003043",
            "extra": "mean: 505.3991404133023 usec\nrounds: 2127"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.672579640568951,
            "unit": "iter/sec",
            "range": "stddev: 0.0008585617966360364",
            "extra": "mean: 73.13908759637603 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 409.94672280656164,
            "unit": "iter/sec",
            "range": "stddev: 0.00005211296630420606",
            "extra": "mean: 2.439341368931646 msec\nrounds: 419"
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
          "id": "b8084dfb36a2d0b650b3b0d65fda0ace894899b4",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-25T13:40:33+01:00",
          "tree_id": "9a3b54b3f654b50b6bc5d9cf0dfd834332021905",
          "url": "https://github.com/equinor/ert/commit/b8084dfb36a2d0b650b3b0d65fda0ace894899b4"
        },
        "date": 1706186452855,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8743022706557921,
            "unit": "iter/sec",
            "range": "stddev: 0.0032827185107066814",
            "extra": "mean: 1.1437691900879146 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8576070180763826,
            "unit": "iter/sec",
            "range": "stddev: 0.012279652389426299",
            "extra": "mean: 1.1660352339968085 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8528986393821053,
            "unit": "iter/sec",
            "range": "stddev: 0.011491133352422411",
            "extra": "mean: 1.172472265549004 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7355.177939930396,
            "unit": "iter/sec",
            "range": "stddev: 0.00000755073800289367",
            "extra": "mean: 135.9586414043252 usec\nrounds: 7847"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8839539934570222,
            "unit": "iter/sec",
            "range": "stddev: 0.004768128144813408",
            "extra": "mean: 530.7985245250165 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.333483890896605,
            "unit": "iter/sec",
            "range": "stddev: 0.0035831024460476484",
            "extra": "mean: 428.5437769256532 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.8827210142742716,
            "unit": "iter/sec",
            "range": "stddev: 0.007824585464412226",
            "extra": "mean: 531.1461403034627 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 6741.139591870939,
            "unit": "iter/sec",
            "range": "stddev: 0.00003686376441411499",
            "extra": "mean: 148.34287087095603 usec\nrounds: 7728"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.850706262965142,
            "unit": "iter/sec",
            "range": "stddev: 0.006115995521050086",
            "extra": "mean: 1.1754938731901348 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8559834506439836,
            "unit": "iter/sec",
            "range": "stddev: 0.012288826113356157",
            "extra": "mean: 1.1682468852028252 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8538809708025263,
            "unit": "iter/sec",
            "range": "stddev: 0.019326532306730017",
            "extra": "mean: 1.1711234167218207 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 6935.581555319143,
            "unit": "iter/sec",
            "range": "stddev: 0.000020141084391849746",
            "extra": "mean: 144.18401572007537 usec\nrounds: 7633"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.873533922042719,
            "unit": "iter/sec",
            "range": "stddev: 0.004680542148690234",
            "extra": "mean: 533.7506773881614 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.2712481256528867,
            "unit": "iter/sec",
            "range": "stddev: 0.006629035417297145",
            "extra": "mean: 440.2865493670106 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8849322985909958,
            "unit": "iter/sec",
            "range": "stddev: 0.0033448578048652184",
            "extra": "mean: 530.5230329744518 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7063.747351099618,
            "unit": "iter/sec",
            "range": "stddev: 0.00001091592367955085",
            "extra": "mean: 141.5679171827018 usec\nrounds: 7675"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 707.770814283344,
            "unit": "iter/sec",
            "range": "stddev: 0.0000874927906527994",
            "extra": "mean: 1.4128867421759315 msec\nrounds: 753"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1635.91574745657,
            "unit": "iter/sec",
            "range": "stddev: 0.000041294678817471866",
            "extra": "mean: 611.2784240598845 usec\nrounds: 1826"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.360840216982249,
            "unit": "iter/sec",
            "range": "stddev: 0.0014703315637845634",
            "extra": "mean: 74.84559232502112 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 361.80139608143116,
            "unit": "iter/sec",
            "range": "stddev: 0.00021948957723229068",
            "extra": "mean: 2.7639473225662416 msec\nrounds: 374"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.0827672522944073,
            "unit": "iter/sec",
            "range": "stddev: 0.03507637182006935",
            "extra": "mean: 12.082073190528899 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.1043521566037165,
            "unit": "iter/sec",
            "range": "stddev: 0.09460905573295703",
            "extra": "mean: 9.582935633976012 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.07935603696403135,
            "unit": "iter/sec",
            "range": "stddev: 0.2737160148928843",
            "extra": "mean: 12.601435735169797 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7026.0049137315455,
            "unit": "iter/sec",
            "range": "stddev: 0.000019212862708510307",
            "extra": "mean: 142.32839462517472 usec\nrounds: 7718"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.21103298729514935,
            "unit": "iter/sec",
            "range": "stddev: 0.026748025274893915",
            "extra": "mean: 4.738595670834184 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.8072680039012647,
            "unit": "iter/sec",
            "range": "stddev: 0.007595068697656804",
            "extra": "mean: 1.2387459866702557 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21351959047870217,
            "unit": "iter/sec",
            "range": "stddev: 0.02215338153634732",
            "extra": "mean: 4.683411005791276 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 6549.655759017477,
            "unit": "iter/sec",
            "range": "stddev: 0.000016455278296359265",
            "extra": "mean: 152.6797799446503 usec\nrounds: 7666"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08116793411403964,
            "unit": "iter/sec",
            "range": "stddev: 0.08726335644245958",
            "extra": "mean: 12.320136158633977 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10450050599999393,
            "unit": "iter/sec",
            "range": "stddev: 0.031331834339170694",
            "extra": "mean: 9.569331654720008 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08193347145684461,
            "unit": "iter/sec",
            "range": "stddev: 0.13436109580176295",
            "extra": "mean: 12.205024176556616 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7000.775566448165,
            "unit": "iter/sec",
            "range": "stddev: 0.000013119664807452562",
            "extra": "mean: 142.84131672390532 usec\nrounds: 7726"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.21737347800255358,
            "unit": "iter/sec",
            "range": "stddev: 0.012693908753699697",
            "extra": "mean: 4.600377236399799 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.8050552327510121,
            "unit": "iter/sec",
            "range": "stddev: 0.007201455697905933",
            "extra": "mean: 1.242150798253715 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.2134193634576419,
            "unit": "iter/sec",
            "range": "stddev: 0.030253756083156447",
            "extra": "mean: 4.6856104516424235 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7221.040751235516,
            "unit": "iter/sec",
            "range": "stddev: 0.000009978272411270602",
            "extra": "mean: 138.48419285390412 usec\nrounds: 7829"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 739.9315039386025,
            "unit": "iter/sec",
            "range": "stddev: 0.000046983987298237435",
            "extra": "mean: 1.3514764470455325 msec\nrounds: 771"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2059.8958244015107,
            "unit": "iter/sec",
            "range": "stddev: 0.00003484580674875957",
            "extra": "mean: 485.46144331864144 usec\nrounds: 2220"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.124624160767423,
            "unit": "iter/sec",
            "range": "stddev: 0.001392172449017306",
            "extra": "mean: 76.19265799543687 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 392.1557360950832,
            "unit": "iter/sec",
            "range": "stddev: 0.00003906007877725167",
            "extra": "mean: 2.5500073260627687 msec\nrounds: 400"
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
          "id": "aca29f28e60edcebdbe1b09f95c752e3e5e95947",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-25T16:40:36+01:00",
          "tree_id": "b752f7d07cd14699b890feca602a5ed80923e6a7",
          "url": "https://github.com/equinor/ert/commit/aca29f28e60edcebdbe1b09f95c752e3e5e95947"
        },
        "date": 1706197256314,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8747435182780381,
            "unit": "iter/sec",
            "range": "stddev: 0.017970955710983165",
            "extra": "mean: 1.1431922376155854 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8556665026366261,
            "unit": "iter/sec",
            "range": "stddev: 0.015494795820468105",
            "extra": "mean: 1.1686796163208784 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8746093226944451,
            "unit": "iter/sec",
            "range": "stddev: 0.013682514217631475",
            "extra": "mean: 1.1433676431886852 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7042.794518064061,
            "unit": "iter/sec",
            "range": "stddev: 0.000015601611513933497",
            "extra": "mean: 141.98909217571241 usec\nrounds: 7826"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8887124584170973,
            "unit": "iter/sec",
            "range": "stddev: 0.012116674702930494",
            "extra": "mean: 529.4612186960876 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.267954733134185,
            "unit": "iter/sec",
            "range": "stddev: 0.010183494989700819",
            "extra": "mean: 440.9259079955518 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.9504005113690275,
            "unit": "iter/sec",
            "range": "stddev: 0.0016550476931708768",
            "extra": "mean: 512.7152060158551 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7260.955554436937,
            "unit": "iter/sec",
            "range": "stddev: 0.000008709145279985791",
            "extra": "mean: 137.72291986953869 usec\nrounds: 7826"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8573426810529762,
            "unit": "iter/sec",
            "range": "stddev: 0.0123301609348432",
            "extra": "mean: 1.1663947475142777 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8613220347767042,
            "unit": "iter/sec",
            "range": "stddev: 0.009454762313742253",
            "extra": "mean: 1.161005941592157 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.857469081124878,
            "unit": "iter/sec",
            "range": "stddev: 0.01608131753147131",
            "extra": "mean: 1.166222808510065 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 6957.099432825968,
            "unit": "iter/sec",
            "range": "stddev: 0.000015283877356579165",
            "extra": "mean: 143.73806349261864 usec\nrounds: 7847"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.9107045958006825,
            "unit": "iter/sec",
            "range": "stddev: 0.004563270402537678",
            "extra": "mean: 523.3671401627362 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.3021552673935077,
            "unit": "iter/sec",
            "range": "stddev: 0.008840176626318724",
            "extra": "mean: 434.37556717544794 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.9580637526082112,
            "unit": "iter/sec",
            "range": "stddev: 0.003062103761667961",
            "extra": "mean: 510.7086011208594 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7145.368647413563,
            "unit": "iter/sec",
            "range": "stddev: 0.000010061238037699172",
            "extra": "mean: 139.95079181281625 usec\nrounds: 7850"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 770.8871029924312,
            "unit": "iter/sec",
            "range": "stddev: 0.000010711963818793855",
            "extra": "mean: 1.2972068103334429 msec\nrounds: 784"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1922.4296479843408,
            "unit": "iter/sec",
            "range": "stddev: 0.00001529695660491269",
            "extra": "mean: 520.1750821147061 usec\nrounds: 2110"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.684992885661284,
            "unit": "iter/sec",
            "range": "stddev: 0.0006740571714752841",
            "extra": "mean: 73.0727453316961 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 393.60720115895157,
            "unit": "iter/sec",
            "range": "stddev: 0.00003266310089002011",
            "extra": "mean: 2.5406039245612453 msec\nrounds: 401"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08320014153046079,
            "unit": "iter/sec",
            "range": "stddev: 0.039146149858904036",
            "extra": "mean: 12.019210323505103 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.09870570618046354,
            "unit": "iter/sec",
            "range": "stddev: 0.2258850486455635",
            "extra": "mean: 10.131126544717699 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.07418747563598463,
            "unit": "iter/sec",
            "range": "stddev: 1.2854339905756287",
            "extra": "mean: 13.479364157188684 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 6944.531119403357,
            "unit": "iter/sec",
            "range": "stddev: 0.000024823236528365724",
            "extra": "mean: 143.99820273048405 usec\nrounds: 7779"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.20885202588846136,
            "unit": "iter/sec",
            "range": "stddev: 0.02379955230742679",
            "extra": "mean: 4.788079003524035 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7979806871045112,
            "unit": "iter/sec",
            "range": "stddev: 0.014636921466952783",
            "extra": "mean: 1.2531631606630982 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.2071022829358174,
            "unit": "iter/sec",
            "range": "stddev: 0.0642905790955365",
            "extra": "mean: 4.828531997930258 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 7189.428717597659,
            "unit": "iter/sec",
            "range": "stddev: 0.000009574640552791297",
            "extra": "mean: 139.093110075949 usec\nrounds: 7752"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.0685079649031662,
            "unit": "iter/sec",
            "range": "stddev: 0.7867185455414721",
            "extra": "mean: 14.596842884086072 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.08682001002219582,
            "unit": "iter/sec",
            "range": "stddev: 0.27351777122878385",
            "extra": "mean: 11.518082061316818 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.0797501236951477,
            "unit": "iter/sec",
            "range": "stddev: 0.09762169326921302",
            "extra": "mean: 12.539165504276752 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7078.076852099309,
            "unit": "iter/sec",
            "range": "stddev: 0.000014286594906760544",
            "extra": "mean: 141.28131424617223 usec\nrounds: 7885"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.20756338755585949,
            "unit": "iter/sec",
            "range": "stddev: 0.016652166247353932",
            "extra": "mean: 4.81780535466969 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.7954621388294449,
            "unit": "iter/sec",
            "range": "stddev: 0.039304200187901824",
            "extra": "mean: 1.2571308566257358 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.20612424396456644,
            "unit": "iter/sec",
            "range": "stddev: 0.05479651137222464",
            "extra": "mean: 4.8514429004862905 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7343.042269084153,
            "unit": "iter/sec",
            "range": "stddev: 0.00002714373564053465",
            "extra": "mean: 136.183337008725 usec\nrounds: 7892"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 774.7924389954615,
            "unit": "iter/sec",
            "range": "stddev: 0.000013794962241370248",
            "extra": "mean: 1.2906682482556566 msec\nrounds: 788"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2047.6469631286907,
            "unit": "iter/sec",
            "range": "stddev: 0.000022132341916200903",
            "extra": "mean: 488.3654350611571 usec\nrounds: 2178"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.009735186920398,
            "unit": "iter/sec",
            "range": "stddev: 0.0011270532926925334",
            "extra": "mean: 76.86551537231675 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 393.75175617370786,
            "unit": "iter/sec",
            "range": "stddev: 0.00032207197739462986",
            "extra": "mean: 2.5396712124347687 msec\nrounds: 405"
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
          "id": "c41ee75a97d90b382a6596e6e39f15fae671a00a",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-25T19:40:10+01:00",
          "tree_id": "b4efa52c4a11fe71316b26cf520a3e9d828d78f2",
          "url": "https://github.com/equinor/ert/commit/c41ee75a97d90b382a6596e6e39f15fae671a00a"
        },
        "date": 1706208033001,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8463074527976249,
            "unit": "iter/sec",
            "range": "stddev: 0.015182408996514771",
            "extra": "mean: 1.1816036792472004 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8331304523997988,
            "unit": "iter/sec",
            "range": "stddev: 0.017631317045848586",
            "extra": "mean: 1.2002922196872532 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8432990517468082,
            "unit": "iter/sec",
            "range": "stddev: 0.032653137389528616",
            "extra": "mean: 1.1858189546503126 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 6994.801577239549,
            "unit": "iter/sec",
            "range": "stddev: 0.000016984345036177542",
            "extra": "mean: 142.9633119621162 usec\nrounds: 7985"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8561694344234114,
            "unit": "iter/sec",
            "range": "stddev: 0.013925675155776068",
            "extra": "mean: 538.7439214624465 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.225895921570951,
            "unit": "iter/sec",
            "range": "stddev: 0.013449868034577879",
            "extra": "mean: 449.2573036812246 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.845487604923973,
            "unit": "iter/sec",
            "range": "stddev: 0.002543483686030858",
            "extra": "mean: 541.8622142635286 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7108.181389379062,
            "unit": "iter/sec",
            "range": "stddev: 0.000015021929524837667",
            "extra": "mean: 140.68296027084858 usec\nrounds: 8034"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.837428727461051,
            "unit": "iter/sec",
            "range": "stddev: 0.011337862300994556",
            "extra": "mean: 1.194131473172456 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.8257591170214521,
            "unit": "iter/sec",
            "range": "stddev: 0.022663074689766707",
            "extra": "mean: 1.211006913986057 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8316061830843982,
            "unit": "iter/sec",
            "range": "stddev: 0.01890655937567185",
            "extra": "mean: 1.2024922617711127 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 7021.863583730082,
            "unit": "iter/sec",
            "range": "stddev: 0.000016493818827789894",
            "extra": "mean: 142.4123365650448 usec\nrounds: 8016"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8207011383508165,
            "unit": "iter/sec",
            "range": "stddev: 0.007258957388562715",
            "extra": "mean: 549.2389601655304 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.1647059568088385,
            "unit": "iter/sec",
            "range": "stddev: 0.018420406416246884",
            "extra": "mean: 461.95650584995747 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.844403167289062,
            "unit": "iter/sec",
            "range": "stddev: 0.006843922807811471",
            "extra": "mean: 542.1808082610369 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 6939.5916461672705,
            "unit": "iter/sec",
            "range": "stddev: 0.000019190686403028526",
            "extra": "mean: 144.10069799313035 usec\nrounds: 7952"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 729.7449681881048,
            "unit": "iter/sec",
            "range": "stddev: 0.00004046056463533336",
            "extra": "mean: 1.3703417544391099 msec\nrounds: 763"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1841.4007083714012,
            "unit": "iter/sec",
            "range": "stddev: 0.00004811020076436457",
            "extra": "mean: 543.0648502815201 usec\nrounds: 2075"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.149863445949508,
            "unit": "iter/sec",
            "range": "stddev: 0.00280706025348951",
            "extra": "mean: 76.04641706815788 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 378.99167258396574,
            "unit": "iter/sec",
            "range": "stddev: 0.00006486445578003793",
            "extra": "mean: 2.6385804025244113 msec\nrounds: 395"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08143530368436717,
            "unit": "iter/sec",
            "range": "stddev: 0.08570323873411571",
            "extra": "mean: 12.279686508886517 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10211733793749803,
            "unit": "iter/sec",
            "range": "stddev: 0.07036880868539541",
            "extra": "mean: 9.792656371556223 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.07888350165209618,
            "unit": "iter/sec",
            "range": "stddev: 0.09197725875504406",
            "extra": "mean: 12.67692203130573 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 6843.830117879102,
            "unit": "iter/sec",
            "range": "stddev: 0.000019292914945982938",
            "extra": "mean: 146.11701090995217 usec\nrounds: 7963"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.20441621331230164,
            "unit": "iter/sec",
            "range": "stddev: 0.02571198083435467",
            "extra": "mean: 4.891979866940528 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7861022800501654,
            "unit": "iter/sec",
            "range": "stddev: 0.01634161255459679",
            "extra": "mean: 1.272099096234888 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.20687627798618033,
            "unit": "iter/sec",
            "range": "stddev: 0.06009758305540088",
            "extra": "mean: 4.833806996792555 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 6923.350626828493,
            "unit": "iter/sec",
            "range": "stddev: 0.000018318344242971388",
            "extra": "mean: 144.43873406106667 usec\nrounds: 8068"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08022612538801653,
            "unit": "iter/sec",
            "range": "stddev: 0.12790980395687424",
            "extra": "mean: 12.464767495170236 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10233923503782243,
            "unit": "iter/sec",
            "range": "stddev: 0.09640120210455207",
            "extra": "mean: 9.77142343921587 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.07921570164761335,
            "unit": "iter/sec",
            "range": "stddev: 0.07179447932499687",
            "extra": "mean: 12.623759926389903 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 6757.996720014237,
            "unit": "iter/sec",
            "range": "stddev: 0.00001988287573719766",
            "extra": "mean: 147.9728448281776 usec\nrounds: 7902"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.20756674799126054,
            "unit": "iter/sec",
            "range": "stddev: 0.04517134254933867",
            "extra": "mean: 4.817727356031537 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.782952861965219,
            "unit": "iter/sec",
            "range": "stddev: 0.025903056672832752",
            "extra": "mean: 1.2772160989232362 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.21088012207507947,
            "unit": "iter/sec",
            "range": "stddev: 0.055978974563126824",
            "extra": "mean: 4.742030638828874 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 6646.189318242917,
            "unit": "iter/sec",
            "range": "stddev: 0.000016373811342915738",
            "extra": "mean: 150.462159910963 usec\nrounds: 7914"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 725.2897587647096,
            "unit": "iter/sec",
            "range": "stddev: 0.00006268413042839604",
            "extra": "mean: 1.3787592998737057 msec\nrounds: 745"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 1904.9526076760746,
            "unit": "iter/sec",
            "range": "stddev: 0.00005723604595581478",
            "extra": "mean: 524.9474427712606 usec\nrounds: 2206"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 12.512113952243942,
            "unit": "iter/sec",
            "range": "stddev: 0.0023153998229135317",
            "extra": "mean: 79.92254576778838 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 406.348479063493,
            "unit": "iter/sec",
            "range": "stddev: 0.00009648833644158908",
            "extra": "mean: 2.460941904605351 msec\nrounds: 423"
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
          "id": "92deaae5d644a27c9dbe1f51fabf759e8e171f0e",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-26T10:39:18+01:00",
          "tree_id": "7dc6fa51603bb3b2bf8cfd41fb44e8eb07440e18",
          "url": "https://github.com/equinor/ert/commit/92deaae5d644a27c9dbe1f51fabf759e8e171f0e"
        },
        "date": 1706261977409,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8800501784032906,
            "unit": "iter/sec",
            "range": "stddev: 0.005880361299183128",
            "extra": "mean: 1.1362988435663284 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8492063762307664,
            "unit": "iter/sec",
            "range": "stddev: 0.019932406433996363",
            "extra": "mean: 1.1775700559839606 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8663299409687174,
            "unit": "iter/sec",
            "range": "stddev: 0.0070661462136095065",
            "extra": "mean: 1.15429463153705 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7016.712076576813,
            "unit": "iter/sec",
            "range": "stddev: 0.000028215057492840648",
            "extra": "mean: 142.51689239725252 usec\nrounds: 7924"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.91661710355271,
            "unit": "iter/sec",
            "range": "stddev: 0.0035645754991563933",
            "extra": "mean: 521.7526224441826 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.346507791705159,
            "unit": "iter/sec",
            "range": "stddev: 0.009321615583642572",
            "extra": "mean: 426.1652160435915 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.851639702479616,
            "unit": "iter/sec",
            "range": "stddev: 0.005440328212418271",
            "extra": "mean: 540.0618698447943 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7269.68439401358,
            "unit": "iter/sec",
            "range": "stddev: 0.000012608017908333249",
            "extra": "mean: 137.5575535058272 usec\nrounds: 7969"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8437187501752449,
            "unit": "iter/sec",
            "range": "stddev: 0.015093017928045599",
            "extra": "mean: 1.185229082312435 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.86465894059609,
            "unit": "iter/sec",
            "range": "stddev: 0.013318246703306487",
            "extra": "mean: 1.1565253686159849 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.8785681803604672,
            "unit": "iter/sec",
            "range": "stddev: 0.0051508070416443836",
            "extra": "mean: 1.1382155902683735 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 7295.330917913184,
            "unit": "iter/sec",
            "range": "stddev: 0.000008474015532335079",
            "extra": "mean: 137.0739739227687 usec\nrounds: 7936"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.9125315053624776,
            "unit": "iter/sec",
            "range": "stddev: 0.005613764624468488",
            "extra": "mean: 522.8672035969794 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.3437413567399257,
            "unit": "iter/sec",
            "range": "stddev: 0.004305442812945779",
            "extra": "mean: 426.66824012994766 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.9176861441108177,
            "unit": "iter/sec",
            "range": "stddev: 0.004755735864876393",
            "extra": "mean: 521.4617642574012 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7188.160964703186,
            "unit": "iter/sec",
            "range": "stddev: 0.000011693294266407035",
            "extra": "mean: 139.11764148165426 usec\nrounds: 7820"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 748.7265390655722,
            "unit": "iter/sec",
            "range": "stddev: 0.000016282960358082633",
            "extra": "mean: 1.3356011144576534 msec\nrounds: 775"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 1993.0440694657527,
            "unit": "iter/sec",
            "range": "stddev: 0.00006445040422398195",
            "extra": "mean: 501.7450518633318 usec\nrounds: 2103"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 14.258793758878117,
            "unit": "iter/sec",
            "range": "stddev: 0.0004778975759710431",
            "extra": "mean: 70.13215962797403 msec\nrounds: 15"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 405.1961926001724,
            "unit": "iter/sec",
            "range": "stddev: 0.00003685945149194845",
            "extra": "mean: 2.46794026760945 msec\nrounds: 412"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08240772763748494,
            "unit": "iter/sec",
            "range": "stddev: 0.08958828607876139",
            "extra": "mean: 12.134784305654467 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10332277696499637,
            "unit": "iter/sec",
            "range": "stddev: 0.08701795359298983",
            "extra": "mean: 9.678408085554839 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.08179243982954114,
            "unit": "iter/sec",
            "range": "stddev: 0.08240439280870004",
            "extra": "mean: 12.226068840641528 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7124.229032378522,
            "unit": "iter/sec",
            "range": "stddev: 0.000013758946773832638",
            "extra": "mean: 140.36606564094924 usec\nrounds: 7779"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.210734621330278,
            "unit": "iter/sec",
            "range": "stddev: 0.028559699747198512",
            "extra": "mean: 4.745304751954973 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.8063017725335476,
            "unit": "iter/sec",
            "range": "stddev: 0.018515677479522315",
            "extra": "mean: 1.2402304373681545 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21641996921257203,
            "unit": "iter/sec",
            "range": "stddev: 0.03853795364205104",
            "extra": "mean: 4.620645699370653 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 7243.321270082453,
            "unit": "iter/sec",
            "range": "stddev: 0.000013428072142098187",
            "extra": "mean: 138.0582142794581 usec\nrounds: 7889"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08185689386743794,
            "unit": "iter/sec",
            "range": "stddev: 0.21792368091154252",
            "extra": "mean: 12.21644204603508 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.10357150008550162,
            "unit": "iter/sec",
            "range": "stddev: 0.22203337549972757",
            "extra": "mean: 9.655165747087448 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.08213140691375165,
            "unit": "iter/sec",
            "range": "stddev: 0.0926708137785991",
            "extra": "mean: 12.175610251631587 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7329.899781789658,
            "unit": "iter/sec",
            "range": "stddev: 0.00000759960031342446",
            "extra": "mean: 136.42751330439629 usec\nrounds: 7806"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.21694243179973483,
            "unit": "iter/sec",
            "range": "stddev: 0.051478185585792285",
            "extra": "mean: 4.609517795592547 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.8254164314798721,
            "unit": "iter/sec",
            "range": "stddev: 0.005887774369627214",
            "extra": "mean: 1.2115096839144825 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.2168036997590643,
            "unit": "iter/sec",
            "range": "stddev: 0.026820598447730146",
            "extra": "mean: 4.612467412278056 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7250.890394793714,
            "unit": "iter/sec",
            "range": "stddev: 0.000014284967314737105",
            "extra": "mean: 137.91409682844196 usec\nrounds: 7992"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 748.492686409891,
            "unit": "iter/sec",
            "range": "stddev: 0.00004591942055113773",
            "extra": "mean: 1.3360183982510927 msec\nrounds: 779"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 2131.0911481140924,
            "unit": "iter/sec",
            "range": "stddev: 0.00005510253246246408",
            "extra": "mean: 469.24318600119443 usec\nrounds: 2236"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 14.165691552475819,
            "unit": "iter/sec",
            "range": "stddev: 0.00037294082551633784",
            "extra": "mean: 70.59309432903925 msec\nrounds: 15"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 398.9484663728503,
            "unit": "iter/sec",
            "range": "stddev: 0.00008720913363859021",
            "extra": "mean: 2.506589407628948 msec\nrounds: 413"
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
          "id": "232b939055f04b58ad5a1e0340d92212340caf24",
          "message": "Update dark storage performance benchmark result",
          "timestamp": "2024-01-26T13:39:59+01:00",
          "tree_id": "64b50aa0e09ec708ba901e0c38a7bf0ddfe02d13",
          "url": "https://github.com/equinor/ert/commit/232b939055f04b58ad5a1e0340d92212340caf24"
        },
        "date": 1706272821802,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_result]",
            "value": 0.8635807482507768,
            "unit": "iter/sec",
            "range": "stddev: 0.010087628245298323",
            "extra": "mean: 1.1579693063162266 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_parquet]",
            "value": 0.8542379443776371,
            "unit": "iter/sec",
            "range": "stddev: 0.013786926806286964",
            "extra": "mean: 1.1706340213306248 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_record_csv]",
            "value": 0.8808183994877343,
            "unit": "iter/sec",
            "range": "stddev: 0.008111480267561587",
            "extra": "mean: 1.1353078007698059 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_parameters]",
            "value": 7039.878189826643,
            "unit": "iter/sec",
            "range": "stddev: 0.00002632428058786591",
            "extra": "mean: 142.04791234102657 usec\nrounds: 8009"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_result]",
            "value": 1.8794701877027349,
            "unit": "iter/sec",
            "range": "stddev: 0.010281083909564519",
            "extra": "mean: 532.0648374967277 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_parquet]",
            "value": 2.324268425806875,
            "unit": "iter/sec",
            "range": "stddev: 0.005900539171383591",
            "extra": "mean: 430.2429052069783 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_record_csv]",
            "value": 1.9149879591355006,
            "unit": "iter/sec",
            "range": "stddev: 0.0021661822518754363",
            "extra": "mean: 522.1964948810637 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_parameters]",
            "value": 7114.6332812573555,
            "unit": "iter/sec",
            "range": "stddev: 0.000027972472790391702",
            "extra": "mean: 140.55538219157123 usec\nrounds: 7984"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_result]",
            "value": 0.8638822103712794,
            "unit": "iter/sec",
            "range": "stddev: 0.022107521193819066",
            "extra": "mean: 1.1575652189552783 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_parquet]",
            "value": 0.851990375381633,
            "unit": "iter/sec",
            "range": "stddev: 0.004232269527236292",
            "extra": "mean: 1.1737221791408956 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_record_csv]",
            "value": 0.859349155502611,
            "unit": "iter/sec",
            "range": "stddev: 0.011688750773500801",
            "extra": "mean: 1.1636713594198227 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_parameters]",
            "value": 6667.43455514305,
            "unit": "iter/sec",
            "range": "stddev: 0.00001711226783579378",
            "extra": "mean: 149.9827244991301 usec\nrounds: 7550"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_result]",
            "value": 1.8801460628702726,
            "unit": "iter/sec",
            "range": "stddev: 0.021431350280244076",
            "extra": "mean: 531.8735707551241 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_parquet]",
            "value": 2.22569331961507,
            "unit": "iter/sec",
            "range": "stddev: 0.014977161616101842",
            "extra": "mean: 449.2981989867985 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_record_csv]",
            "value": 1.8962520593149532,
            "unit": "iter/sec",
            "range": "stddev: 0.003815893164187888",
            "extra": "mean: 527.3560522124171 msec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_parameters]",
            "value": 7438.320534147203,
            "unit": "iter/sec",
            "range": "stddev: 0.000007400322009955066",
            "extra": "mean: 134.4389496808165 usec\nrounds: 7949"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary-get_observations]",
            "value": 770.8884453106754,
            "unit": "iter/sec",
            "range": "stddev: 0.000021769995171312373",
            "extra": "mean: 1.2972045515573793 msec\nrounds: 775"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data-get_observations]",
            "value": 2108.06968714243,
            "unit": "iter/sec",
            "range": "stddev: 0.00001031729592059238",
            "extra": "mean: 474.3676198653274 usec\nrounds: 2176"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-summary_with_obs-get_observations]",
            "value": 13.760364032686574,
            "unit": "iter/sec",
            "range": "stddev: 0.0008571258385163425",
            "extra": "mean: 72.67249599099159 msec\nrounds: 15"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 100-gen_data_with_obs-get_observations]",
            "value": 405.27679161808584,
            "unit": "iter/sec",
            "range": "stddev: 0.000051029796864165435",
            "extra": "mean: 2.4674494584490145 msec\nrounds: 420"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_result]",
            "value": 0.08166068461882889,
            "unit": "iter/sec",
            "range": "stddev: 0.27676587426958876",
            "extra": "mean: 12.245794958341866 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_parquet]",
            "value": 0.10374931722583453,
            "unit": "iter/sec",
            "range": "stddev: 0.06369027795582398",
            "extra": "mean: 9.6386176481843 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_record_csv]",
            "value": 0.08200445040807164,
            "unit": "iter/sec",
            "range": "stddev: 0.04930518728861782",
            "extra": "mean: 12.194460118003189 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_parameters]",
            "value": 7447.4443714115205,
            "unit": "iter/sec",
            "range": "stddev: 0.000006400639541158113",
            "extra": "mean: 134.2742490079814 usec\nrounds: 7881"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_result]",
            "value": 0.2138206646871383,
            "unit": "iter/sec",
            "range": "stddev: 0.0244540676614095",
            "extra": "mean: 4.67681644083932 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_parquet]",
            "value": 0.7929792084801325,
            "unit": "iter/sec",
            "range": "stddev: 0.02287160902495188",
            "extra": "mean: 1.2610671116039156 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_record_csv]",
            "value": 0.21599860935900272,
            "unit": "iter/sec",
            "range": "stddev: 0.02419004468476583",
            "extra": "mean: 4.62965943608433 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_parameters]",
            "value": 6564.467502027083,
            "unit": "iter/sec",
            "range": "stddev: 0.000028837699391654537",
            "extra": "mean: 152.33528076591188 usec\nrounds: 7707"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_result]",
            "value": 0.08066294118943118,
            "unit": "iter/sec",
            "range": "stddev: 0.17068255375834926",
            "extra": "mean: 12.397266765311361 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_parquet]",
            "value": 0.1042414588658865,
            "unit": "iter/sec",
            "range": "stddev: 0.1051786304070574",
            "extra": "mean: 9.593112096469849 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_record_csv]",
            "value": 0.07982412566762728,
            "unit": "iter/sec",
            "range": "stddev: 0.07938023876635132",
            "extra": "mean: 12.527540911175311 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_parameters]",
            "value": 7316.555147682161,
            "unit": "iter/sec",
            "range": "stddev: 0.000009100389253361558",
            "extra": "mean: 136.6763428711111 usec\nrounds: 7976"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_result]",
            "value": 0.20124865286195104,
            "unit": "iter/sec",
            "range": "stddev: 0.04876362178999072",
            "extra": "mean: 4.968977360986173 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_parquet]",
            "value": 0.7754427217820089,
            "unit": "iter/sec",
            "range": "stddev: 0.026489257583600923",
            "extra": "mean: 1.2895858996547758 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_record_csv]",
            "value": 0.20867956173702942,
            "unit": "iter/sec",
            "range": "stddev: 0.052187327250351793",
            "extra": "mean: 4.7920361327007415 sec\nrounds: 5"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_parameters]",
            "value": 7084.917630347134,
            "unit": "iter/sec",
            "range": "stddev: 0.000017888418474568778",
            "extra": "mean: 141.1449013488395 usec\nrounds: 7937"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary-get_observations]",
            "value": 718.9698662354594,
            "unit": "iter/sec",
            "range": "stddev: 0.00017859209749849359",
            "extra": "mean: 1.3908788767963531 msec\nrounds: 759"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data-get_observations]",
            "value": 1951.837481916495,
            "unit": "iter/sec",
            "range": "stddev: 0.00006190079865703134",
            "extra": "mean: 512.3377377803541 usec\nrounds: 2108"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-summary_with_obs-get_observations]",
            "value": 13.632031062328956,
            "unit": "iter/sec",
            "range": "stddev: 0.0013010647481432958",
            "extra": "mean: 73.35664035885462 msec\nrounds: 14"
          },
          {
            "name": "tests/performance_tests/test_dark_storage_performance.py::test_direct_dark_performance_with_libres_facade[gen_x: 2000, sum_x: 2000 reals: 1000-gen_data_with_obs-get_observations]",
            "value": 395.9040859816974,
            "unit": "iter/sec",
            "range": "stddev: 0.00008499280878180661",
            "extra": "mean: 2.5258643075642064 msec\nrounds: 410"
          }
        ]
      }
    ]
  }
}