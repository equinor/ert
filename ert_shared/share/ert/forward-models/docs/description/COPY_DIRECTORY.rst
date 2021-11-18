The job copies the directory :code:`<FROM>` to the target :code:`<TO>`. If
:code:`<TO>` points to a non-existing directory structure, it will be
created first. If the :code:`<TO>` folder already exist it creates a new directory within the existing one.
E.g. :code:`COPY_DIRECTORY (<FROM>=foo, <TO>=bar)` creates :code:`bar/foo` if the directory
:code:`bar` already exists. If :code:`bar` does not exist it becomes a copy of :code:`foo`.
