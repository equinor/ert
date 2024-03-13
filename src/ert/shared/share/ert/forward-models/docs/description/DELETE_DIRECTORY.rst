The :code:`DELETE_DIRECTORY` job will recursively remove a directory
and all the files in the directory. Like the :code:`DELETE_FILE` job
it will *only* delete files and directories which are owned by the
current user. If one delete operation fails the job will continue, but
unless all delete calls succeed (parts of) the directory structure
will remain.

.. warning::
  If the directory to delete is a symlink to a directory, it will only delete
  the link and not the directory. However, if you add a trailing slash to the
  directory name (the symlink), then the link itself is kept, but the directory
  it links to will be removed.
