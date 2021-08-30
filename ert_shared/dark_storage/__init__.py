"""
Dark Storage is an implementation of a subset of ERT Storage's API
(https://github.com/Equinor/ert-storage), where the data is provided by the
legacy EnKFMain and 'storage/' directory.

The purpose of this is to provide users with an API that they can use without
requiring a dedicated PostgreSQL server, and that works with their existing
data. It should be noted that it's unfeasible to implement the entire ERT
Storage API, nor is it possible to have the same guarantees of data integrity
that ERT Storage has.
"""
