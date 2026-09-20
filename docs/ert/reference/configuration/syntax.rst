The ert configuration file format
=================================

The ert config file, workflows, workflow job configuration files and forward
model configuration files all share the same syntax. These are lines starting
with one keyword followed by arguments separated by whitespace. e.g.

::

        QUEUE_SYSTEM LOCAL
        NUM_REALIZATIONS 12
        -- A comment
        FORWARD_MODEL DELETE_FILE(<FILES>="file1 file-- file3")

        FORWARD_MODEL TEMPLATE_RENDER( \
            <INPUT_FILES>=parameters.json, \
            <TEMPLATE_FILE>=<CONFIG_PATH>/resources/SPE1.DATA.jinja2, \
            <OUTPUT_FILE>=SPE1.DATA \
        )

So for example :code:`QUEUE_SYSTEM` is a keyword and :code:`LOCAL` is an
argument. A comment is started with :code:`--`. :code:`FORWARD_MODEL` is a
special keyword that starts an installed forward model step (e.g.
:code:`DELETE_FILE`) with named forward model arguments inside parenthesis,
(e.g. :code:`<FILES>` is set to :code:`file1 file-- file3`. Note that arguments
can be surrounded by quotes (:code:`"`) to make spaces and :code:`--` part of
the argument.

As can be seen, you can have multi-line statements by escaping the newline with
:code:`\\` (like with bash). NB: the newline has to follow directly after
:code:`\\` without any spaces, tabs or other trailing whitespace.
