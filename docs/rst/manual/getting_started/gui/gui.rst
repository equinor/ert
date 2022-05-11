Graphical user interface
========================

Restarting ES-MDA
-----------------
It is possible to restart ES-MDA runs from an intermediate iteration. Note
that this requires a bit of care due to the lack of metadata in current storage.
We are aiming at resolving this in the future in the new storage that soon will
be the standard.

After selecting the ES-MDA algorithm, you first need to set `Current case` to
the case you intend to be your first case to reevaluate. After which the
iteration number will be wrongly injected into the `Target case format`, which
you have to remove manually (reset it to the original `Target case format`).
After which you have to set `Start iteration` to the iteration for which you
are to start from; this number must correspond to the iteration number of the
case you have selected as your `Current case`. We recognize that this induces
some manual checking, but this is due to the lack of metadata mentioned above.
We still hope that this can aid our users and prevent the need of a complete
restart due to cluster issues etc.

.. image:: restart-es-mda.png
