Storage Migration
=================

To keep track of the structural changes done within storage, we utilize a simple versioning scheme where the storage version number indicates which version storage currently has. Every increase of this version number indicates a structural change.

There is no differentiation between the impact of the changes, only that there are changes.

We guarantee that old storages can be migrated to work with newer ERT versions (with a newer storage version) by adding storage migration scripts (for example `to10.py`, `to11.py`).

Migration Process
-----------------

For example, if the local storage version is 6, but ERT is already at version 8, the migration will occur as follows:

**Migration Steps**:

- Step 1: Upgrade to version 7
- Step 2: Upgrade to version 8

Thus, the migration path will be: :code:`6 → 7 → 8`

Important Notes
---------------

- We currently have no established way or practice for reverting storage migrations
- The storage version and ERT version are not directly related

Ruleset
-------
* Changes to storage content constitutes a storage migration version increment
* Each single migration should increment the storage version
* Since migrations cannot be undone, each storage migration should (if possible) be exhaustively tested with unit tests, ref `test_to12.py`, `test_to13.py`, ...
* Add storage migration script toward new version
* Update ert-testdata with storage content from previous ERT tag
* ERT storage changes pertaining to Everest (storage of everest-specific parameter / response configs) must be migrated and kept up-to-date with the latest ERT version, as they are set to be exposed through the GUI (plotting, experiment inspection).
* Storage migrations should not rely on / import any ERT internals, avoid using `local_storage_get_ert_config`. Reason: Storage migrations are meant to be final, and using ERT code effectively prevents us from modifying the ERT code it uses, as it would also modify the migration.

References
----------
- `Storage Migration scripts <https://github.com/equinor/ert/tree/main/src/ert/storage/migration>`_
- `ert-testdata <https://github.com/equinor/ert-testdata/tree/main/all_data_types>`_
