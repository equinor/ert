Forward models
==============

Default jobs
~~~~~~~~~~~~

Reservoir simulation: eclipse/flow
..................................

.. code-block:: bash

    EXECUTABLE script/ecl100
    DEFAULT <VERSION> version
    DEFAULT <NUM_CPU> 1
    DEFAULT OPTS ""
    ARGLIST <ECLBASE> "--version=<VERSION>" "--num-cpu=<NUM_CPU>" OPTS

There are forward models for each of the simulators eclipse100, eclipse300 and
flow, and they are all configured the same way. The version, number of cpu and
whether or not to ignore errors can be configured in the configuration file
when adding the job, as such:

.. code-block:: bash

    FORWARD_MODEL ECLIPSE100(<ECLBASE>, <VERSION>=xxxx, <OPTS>="--ignore-errors")

The :code:`OPTS` argument is optional and can be removed, thus running eclipse
without ignoring errors

Reservoir modelling: RMS
........................

File system utilities
.....................

There are many small jobs for copying and moving files and directories. The jobs
are thin wrappers around corresponding operations in standard libraries, however
additional error checking and "careful" default behaviour have been added in
order to make them *safer* to use. The jobs are typically quite strict, i.e. if
you try to remove a directory with the :code:`DELETE_FILE` job, or visa versa,
the job will fail. The actions performed will typically be logged to the
:code:`xxx.stdout` files:

.. code-block:: bash

    Copying file 'file.txt' to 'target_backup.txt'

And error messages will go to the :code:`xxx.stderr` files.

COPY_FILE
.........

.. code-block:: bash

    STDOUT      copy_file.stdout
    STDERR      copy_file.stderr

    EXECUTABLE  script/copy_file.py
    ARGLIST     <FROM> <TO>

The :code:`COPY_FILE` job will copy a file. If the :code:`<TO>`
argument has a directory component, that directory will be created,
i.e. with the :code:`FORWARD_MODEL`:

.. code-block:: bash

    FORWARD_MODEL COPY_FILE(<FROM>=file1, <TO>=path/to/directory/file1)

The directory :code:`path/to/directory` will be created before the
file is copied, this is an extension of the normal :code:`cp` command
which will *not* create directories in this way.

COPY_FOLDER
..............

.. code-block:: bash

    STDERR    COPY_FOLDER.stderr
    STDOUT    COPY_FOLDER.stdout

    PORTABLE_EXE  /bin/cp
    ARGLIST       -rfv <COPYFROM> <COPYTO>

The job copies the directory :code:`<COPYFROM>` to the target :code:`<COPYTO>`. If
:code:`<COPYTO>` points to a non-existing directory structure, the job will fail as the target
directory need to be created first. In such case, user can use a job :code:`COPY_DIRECTORY`.


COPY_DIRECTORY
..............

.. code-block:: bash

    STDERR      copy_directory.stderr
    STDOUT      copy_directory.stdout

    EXECUTABLE  script/copy_directory.py
    ARGLIST     <FROM> <TO>

The job copies the directory :code:`<FROM>` to the target :code:`<TO>`. If
:code:`<TO>` points to a non-existing directory structure, it will be
created first.

CAREFUL_COPY_FILE
.................

.. code-block:: bash

    STDERR      careful_copy_file.stderr
    STDOUT      careful_copy_file.stdout

    EXECUTABLE  script/careful_copy_file.py
    ARGLIST     <FROM> <TO>

The :code:`CAREFUL_COPY_FILE` job will copy a file if the target :code:`<TO>`
does not already exist. If the :code:`<TO>` argument has a directory component,
that directory will be created, i.e. with the :code:`FORWARD_MODEL`:

.. code-block:: bash

    FORWARD_MODEL CAREFUL_COPY_FILE(<FROM>=file1, <TO>=path/to/directory/file1)

The directory :code:`path/to/directory` will be created before the
file is copied, this is an extension of the normal :code:`cp` command
which will *not* create directories in this way.

DELETE_FILE
...........

.. code-block:: bash

    STDERR      delete_file.stderr
    STDOUT      delete_file.stdout

    EXECUTABLE  script/delete_file.py
    ARGLIST     <FILES>

The :code:`DELETE_FILE` job will *only* remove files which are owned
by the current user, *even if* file system permissions would have
allowed the delete operation to proceed. The :code:`DELETE_FILE` will
*not* delete a directory, and if presented with a symbolic link it
will *only* delete the link, and not the target.


DELETE_DIRECTORY
................

.. code-block:: bash

    STDERR      delete_dir.stderr
    STDOUT      delete_dir.stdout

    EXECUTABLE  script/delete_dir.py
    ARGLIST     <DIRECTORY>

The :code:`DELETE_DIRECTORY` job will recursively remove a directory
and all the files in the directory. Like the :code:`DELETE_FILE` job
it will *only* delete files and directories which are owned by the
current user. If one delete operation fails the job will continue, but
unless all delete calls succeed (parts of) the directory structure
will remain.


MOVE_FILE
.........

.. code-block:: bash

    STDERR      move_file.stderr
    STDOUT      move_file.stdout

    EXECUTABLE  script/move_file.py
    ARGLIST     <FROM>  <TO>

The :code:`MOVE_FILE` job will move file to target directory.
If file already exists, this job will move file to the target directory
and then replace the exisitng file.

MAKE_DIRECTORY
..............

.. code-block:: bash

    STDERR      make_directory.stderr
    STDOUT      make_directory.stdout

    EXECUTABLE  script/make_directory.py
    ARGLIST     <DIRECTORY>


Will create the directory :code:`<DIRECTORY>`, with all sub
directories.


MAKE_SYMLINK / SYMLINK
......................

.. code-block:: bash

    STDERR      make_symlink.stderr
    STDOUT      make_symlink.stdout

    EXECUTABLE  script/symlink.py
    ARGLIST     <TARGET> <LINKNAME>

Will create a symbolic link with name :code:`<LINKNAME>` which points to
:code:`<TARGET>`. If :code:`<LINKNAME>` already exists, it will be updated.


STORE_FILE
..........

.. code-block:: bash

    STDOUT      store_file.stdout
    STDERR      store_file.stderr

    EXECUTABLE  script/store_file.py
    ARGLIST     <STORAGE_PATH> <FILE>
