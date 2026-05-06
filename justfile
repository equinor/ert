poly:
    ert gui test-data/ert/poly_example/poly.ert

snake_oil:
    ert gui test-data/ert/snake_oil/snake_oil.ert

heat_equation:
    ert gui test-data/ert/heat_equation/config.ert

pytest_args := env("ERT_PYTEST_ARGS", "--quiet")

rapid-tests:
    OMP_NUM_THREADS=1 pytest -n auto --dist loadgroup tests/ert/unit_tests tests/everest \
    --hypothesis-profile=fast \
    --benchmark-disable \
    -m "not (slow or unreliable or high_utilization)" \
    --timeout=10 --session-timeout=120 \
    -p no:memray

ert-rapid-tests:
    OMP_NUM_THREADS=1 pytest --dist loadgroup tests/ert/unit_tests \
    --hypothesis-profile=fast \
    --ignore=tests/ert/unit_tests/gui \
    --ignore=tests/ert/unit_tests/dark_storage \
    --ignore=tests/ert/unit_tests/config/test_transfer_functions.py \
    --ignore=tests/ert/unit_tests/ensemble_evaluator/test_ensemble_client.py \
    -m "not (slow or unreliable or high_utilization or creates_tmpdir)" \
    -p no:memray -p no:doctest -p no:benchmark -p no:mpl -p no:cov -p no:pytest-qt

continuous_tests:
    fswatch -r -o --event Created --event Updated --event Removed --event Renamed src/ tests/ --exclude="\.egg-info|__pycache__" | while IFS= read -r _; do nice just ert-rapid-tests; done

fuzz:
    OMP_NUM_THREADS=1 pytest {{pytest_args}} -m "fuzzing" --hypothesis-profile=fuzz tests/ert

screenshot-comparison-test:
    rm -rf /tmp/test_docs_screenshots
    pytest --mpl --mpl-results-path=pytest-mpl_results -v -m "mpl_image_compare or screenshot_test" tests

pack_updated_screenshots:
    #!/bin/bash
    staging="updated-screenshots"
    mkdir -p "$staging"
    for test_dir in pytest-mpl_results/*/; do
      # Filenames from pytest-mpl need a lot of massaging in order to
      # reconstruct the directory structure:
      full_dotted_name=$(basename "$test_dir")
      updated_img="${test_dir}/result.png"
      filename="${full_dotted_name##*.}.png"
      path_with_test_func_name="${full_dotted_name%.*}"
      path_dots="${path_with_test_func_name%.*}"
      rel_path="${path_dots//.//}"
      target_dir="$staging/$rel_path/baseline"
      mkdir -p $staging/$rel_path/baseline
      cp "$updated_img" "$target_dir/$filename"
      echo "Mapped $full_dotted_name -> $target_dir/$filename"
    done
    [ -d "/tmp/test_docs_screenshots" ] && cp -r /tmp/test_docs_screenshots/* "$staging"
    find "$staging" -type f -exec ls -l {} +

ert-gui-tests:
    pytest {{pytest_args}} tests/ert/ui_tests/gui -m "not (mpl_image_compare or screenshot_test)"

ert-cli-tests:
    pytest {{pytest_args}} tests/ert/ui_tests/cli

ert-memory-tests:
    _RJEM_MALLOC_CONF="dirty_decay_ms:100,muzzy_decay_ms:100" pytest -n 2 {{pytest_args}} tests/ert -m "memory_test"
    _RJEM_MALLOC_CONF="dirty_decay_ms:100,muzzy_decay_ms:100" pytest -n 2 {{pytest_args}} tests/ert -m "limit_memory" --memray

ert-unit-tests:
    pytest {{pytest_args}} -n 4 --dist loadgroup --benchmark-disable tests/ert/unit_tests tests/ert/performance_tests -m "not (memory_test or limit_memory)"

ert-doc-tests:
    pytest {{pytest_args}} --doctest-modules src/ --ignore src/ert/dark_storage

everest-tests:
    pytest -n 4 --benchmark-disable --dist loadgroup {{pytest_args}} tests/everest

build-everest-docs:
    sphinx-build -n -v -E -W ./docs/everest ./everest_docs

build-ert-docs:
    sphinx-build -n -v -E -W ./docs/ert ./ert_docs

build-docs: build-ert-docs build-everest-docs

check-types:
    mypy src

test-all:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just everest-tests'

ert-tests:
    parallel -j4 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests'

check-all:
    parallel -j8 ::: 'just ert-gui-tests' 'just ert-cli-tests' 'just ert-unit-tests' 'just ert-doc-tests' 'just everest-tests' 'just check-types' 'just build-everest-docs' 'just build-ert-docs'

update-matplotlib-screenshots:
    pytest --mpl --mpl-generate-path={{justfile_directory()}}/.pics {{pytest_args}} -m "mpl_image_compare" tests/ert/ui_tests/gui
    mv {{justfile_directory()}}/.pics/* {{justfile_directory()}}/tests/ert/ui_tests/gui/baseline/
    pytest --mpl --mpl-generate-path={{justfile_directory()}}/.pics {{pytest_args}} -m "mpl_image_compare" tests/ert/unit_tests/gui/plottery
    mv {{justfile_directory()}}/.pics/* {{justfile_directory()}}/tests/ert/unit_tests/gui/plottery/baseline/
    rm -rf .pics

update-ert-snapshots:
    pytest --snapshot-update {{pytest_args}} -m "snapshot_test" tests/ert

update-everest-snapshots:
    pytest --snapshot-update {{pytest_args}} -m "snapshot_test" tests/everest

update-snapshots:
    just "update-everest-snapshots"
    just "update-ert-snapshots"
