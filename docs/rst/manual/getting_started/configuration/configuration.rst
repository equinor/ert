Configuration Guide (todo)
==========================

.. todo::
    How to create an entire config directory (recreate poly_eval?)

    Several pages with link to relevant chapters


The ERT configuration file serves several purposes, which are:

* Defining which simulation model to use (for example: ECLIPSE).
* Defining which observation file to use.
* Defining how to run simulations.
* Defining where to store results.
* Creating a parametrization of the simulation model.

The configuration file is a plain text file, with one statement per line. The
first word on each line is a keyword, which then is followed by a set of
arguments that are unique to the particular keyword. Except for the DEFINE
keyword, ordering of the keywords is not significant. Lines starting with "- -"
are treated as comments.

.. toctree::
   :maxdepth: 1

   poly_new/guide

