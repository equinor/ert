Storage Migration
=================

To keep track of the structural changes done within storage, we utilize a simple versioning scheme where the storage version number indicates which version storage currently has. Every increase of this version number indicates a structural change.

There is no differentiation between the impact of the changes—only that there are changes.

We guarantee forward compatibility by adding storage-migration scripts that will update any given storage version to the next.

Migration Process
-----------------

For example, if the local storage version is 6, but ERT is already at version 8, the migration will occur as follows:

**Migration Steps**:

- Step 1: Upgrade to version 7
- Step 2: Upgrade to version 8

Thus, the migration path will be: :code:`6 → 7 → 8`

Important Notes
---------------

- We currently have no way of reverting changes backwards between storage versions.
- The storage version and ERT version are not directly related

Ruleset
-------
* Changes to storage content constitutes a storage migration version increment
* Each single migration should increment the storage version
* Add storage migration script toward new version
* Update ert-testdata with storage content from previous ERT tag


References
----------
- `Storage Migration scripts <https://github.com/equinor/ert/tree/main/src/ert/storage/migration>`_
- `ert-testdata <https://github.com/equinor/ert-testdata/tree/main/all_data_types>`_
