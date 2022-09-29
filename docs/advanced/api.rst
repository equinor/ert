:orphan:

ERT API
=======

Localization
------------

General overview
~~~~~~~~~~~~~~~~

To create a configuration for localization you must *"program"* your own local
config commands by writing a Python script, and invoking it from a workflow.

.. highlight:: python

**Local config python script example:**

::

 from ert import ErtScript
 from ecl import EclGrid, EclRegion, Ecl3DKW, EclFile, EclInitFile, EclKW, EclTypeEnum

 class LocalConfigJob(ErtScript):

     def run(self):

         # This example can be used with the REEK data set from the ERT tutorial

         # Get the ert object
         ert = self.ert()

         # An update_step
         update_step = {
            "name": "UPDATE_STEP_NAME",
            "observations": [("WOPR:OP_1_10", list(range(10))], # Add some observations from WOPR:OP_1
            "parameters": [("MULTFLT", [0])} # Add some dataset you want to localize here.

        ert.update_configuration = [update_step]

=========================================================================  ===================================================================================
ERT script function                                                        Purpose
=========================================================================  ===================================================================================
:ref:`getObservations                  <all_obs>`                          Get the observations currently imported, use to filter the observations to localize
:ref:`grid                             <ert_grid>`                         Get the underlying grid use to define active cells in a field
:ref:`EclGrid, EclInitFile             <load_file>`                        Loads eclipse file in restart format
:ref:`EclRegion                        <create_eclregion>`                 Creates a new region for use when defining active regions for fields
:ref:`select_active                    <eclregion_select_all>`             Selects or deselects cells in a region
:ref:`select_equal                     <eclregion_select_value_equal>`     Selects or deselects cells in a region equal to given value
:ref:`select_less                      <eclregion_select_value_less>`      Selects or deselects cells in a region equal less than a given value
:ref:`select_more                      <eclregion_select_value_more>`      Selects or deselects cells in a region equal greater than a given value
:ref:`select_box                       <eclregion_select_box>`             Selects or deselects cells in a box
:ref:`select_islice, _jslice,_kslice   <eclregion_select_slice>`           Selects or deselects cells in a slice
:ref:`select_below_plane               <eclregion_select_plane>`           Selects or deselects cells in a half space defined by a plane
:ref:`select_inside_polygon            <eclregion_select_in_polygon>`      Selects or deselects cells in region inside polygon
:ref:`Example create polygon           <create_polygon>`                   Creates a geo-polygon based on coordinate list
:ref:`Example load polygon             <load_polygon>`                     Loads polygon in Irap RMS format from file
:ref:`Load surface from IRAP file      <surface__init>`                    Create a polygon from IRAP file
:ref:`Select polygon from surface      <geo_region__select_polygon>`       Selects the inside of a polygon from a surface
:ref:`Select halfspace from surface    <geo_region__select_halfspace>`     Selects above or below a line from a surface
=========================================================================  ===================================================================================


.. #####################################################################
.. _all_obs:
.. topic:: getObservations

   This function will retrieve ERT's observations

   *Example:*

   ::

      all_obs = ert.getObservations()


.. #####################################################################
.. _ert_grid:
.. topic:: grid

   This function will retrieve ERT's grid

   *Example:*

   ::

      grid = ert.eclConfig.grid

.. #####################################################################
.. _load_file:
.. topic:: EclGrid, EclInitFile

   This function will load an ECLIPSE file in restart format (i.e. *restart
   file* or *INIT file*), the keywords in this file can then subsequently be
   used in ``ECLREGION_SELECT_VALUE_XXX`` commands below.  The ``KEY`` argument
   is a string which will be used later when we refer to the content of this
   file.

   *Example:*

   ::

      # Load Eclipse grid and init file
      ecl_grid = EclGrid("path/to/FULLMODEL.GRDECL")
      refinit_file = EclInitFile(grid , "path/to/somefile.init")


.. #####################################################################
.. _create_eclregion:
.. topic:: EclRegion

   This function will create a new region ``ECLREGION_NAME``, which can
   subsequently be used when defining active regions for fields.  The second
   argument, ``SELECT_ALL``, is a *boolean* value.  If this value is set to true
   the region will start with all cells selected, if set to false the region
   will start with no cells selected.

   *Example:*

   ::

      # Define Eclipse region
      eclreg_poro = EclRegion(ecl_grid, False)


.. #####################################################################
.. _eclregion_select_all:
.. topic:: select_active

   Will select (or deselect) all the cells in the region.


   *Example:*

   ::

      eclreg_poro.select_active()
      eclreg_poro.deselect_active()


.. #####################################################################
.. _eclregion_select_value_equal:
.. topic:: select_equal

   This function will compare an ``ecl_kw`` instance loaded from file with a
   user supplied value, and select (or deselect) all cells which match this
   value.  It is assumed that the ECLIPSE keyword is an INTEGER keyword, for
   float comparisons use the ``ECLREGION_SELECT_VALUE_LESS`` and
   ``ECLREGION_SELECT_VALUE_MORE`` functions.

   *Example:*

   ::

      # Load Eclipse grid
      ecl_grid = EclGrid("path/to/LOCAL.GRDECL")

      with open("path/to/LOCAL.GRDECL","r") as grdecl_file:
          local_kw = Ecl3DKW.read_grdecl(ecl_grid, grdecl_file, "LOCAL",
                                         ecl_type=EclTypeEnum.ECL_INT_TYPE)

      # Define Eclipse region
      eclreg_poro = EclRegion(ecl_grid, False)
      eclreg_poro.select_equal(local_kw, 1)
      print('GRID LOADED: %s' % ecl_grid)
      print(ecl_grid.getDims())
      print(local_kw.header)


.. #####################################################################
.. _eclregion_select_value_less:
.. topic:: select_less

   This function will compare an ``ecl_kw`` instance loaded from disc with a
   numerical value, and select all cells which have numerical below the limiting
   value.  The ``ecl_kw`` value should be a floating point value like e.g.,
   ``PRESSURE`` or ``PORO``.  The arguments are just as for
   ``ECLREGION_SELECT_VALUE_EQUAL``.

   *Example:*

   ::

      eclreg_poro.select_less(local_kw, 1)


.. #####################################################################
.. _eclregion_select_value_more:
.. topic:: select_more

   This function will compare an ``ecl_kw`` instance loaded from disc with a
   numerical value, and select all cells which have numerical above the limiting
   value.  The ``ecl_kw`` value should be a floating point value like e.g.,
   ``PRESSURE`` or ``PORO``.  The arguments are just as for
   ``ECLREGION_SELECT_VALUE_EQUAL``.


   *Example:*

   ::

      eclreg_poro.select_more(local_kw, 1)


.. #####################################################################
.. _eclregion_select_box:
.. topic:: select_box

   This function will select (or deselect) all the cells in the box defined by
   the six coordinates ``i1 i2 j1 j2 k1 k2``.  The coordinates are inclusive,
   and the counting starts at 1.


   *Example:*

   ::

      eclreg_poro.select_box((0,2,4),(1,3,5))


.. #####################################################################
.. _eclregion_select_slice:
.. topic:: select_islice, _jslice,_kslice

   This function will select a slice in the direction given by ``dir``', which
   can ``x``, ``y``, or ``z``.  Depending on the value of ``dir`` the numbers
   ``n1`` and ``n2`` are interpreted as ``(i1 i2)``, ``(j1 j2)``, or ``(k1
   k2)``, respectively.

   The numbers ``n1`` and ``n2`` are inclusive and the counting starts at 1.  It
   is OK to use very high/low values to imply *"the rest of the cells"* in one
   direction.


   *Example:*

   ::

      eclreg_poro.select_kslice(2,3)


.. #####################################################################

.. _eclregion_select_plane:
.. topic:: select_below_plane

   Will select all points which have positive (sign > 0) distance to the plane
   defined by normal vector ``n = (nx,ny,nz)`` and point ``p = (px,py,pz)``. If
   sign < 0 all cells with negative distance to plane will be selected.

   *Example:*

   ::

      eclreg_poro.select_below_plane((1,1,1), (0,0,0))


.. #####################################################################
.. _eclregion_select_in_polygon:
.. topic:: select_inside_polygon

   Well select all the points which are inside the polygon with name
   ``POLYGON_NAME``.  The polygon should have been created with command
   ``CREATE_POLYGON`` or loaded with command ``LOAD_POLYGON`` first.


   *Example:*

   ::

      polygon = [(0,0), (0,1), (1,0)]
      eclreg_poro.select_inside_polygon(polygon)


.. #####################################################################
.. _create_polygon:
.. topic:: Example create polygon

   Will create a ``geo_polygon`` instance based on the coordinate list:

   ``[(x1,y1), (x2,y2), (x3,y3), ..., (xn,yn)]``

   The polygon should not be explicitly closed --- i.e., you should in general
   have

   ``(x1,y1) != (xn,yn).``

   The polygon will be stored under the name ``POLYGON_NAME`` --- which should
   later be used when referring to the polygon in region select operations.


   *Example:*

   ::

      polygon = [(0,0), (0,1), (1,0)]


.. #####################################################################
.. _load_polygon:
.. topic:: Example load polygon

   Will load a polygon instance from the file ``FILENAME`` --- the file should
   be in *irap RMS* format.  The polygon will be stored under the name
   ``POLYGON_NAME`` which can then later be used to refer to the polygon for
   e.g., select operations.


   *Example:*

   ::

      polygon = []
      with open("polygon.ply", "r") as ply_file:
          for line in ply_file:
              xs, ys = map(float, line.split())
              polygon.append(xs, ys)


.. #####################################################################
.. _surface__init:
.. topic:: Load surface from IRAP file

   Will load a surface from an *IRAP file*.  We can also create a surface
   programmatically.  It is also possible to obtain the underlying pointset.


   *Example for creating programmatically:*

   ::

      # values copied from irap surface_small
      nx, ny = 30,20
      xinc, yinc = 50.0, 50.0
      xstart, ystart = 463325.5625, 7336963.5
      angle = -65.0
      s_args = (None, nx, ny, xinc, yinc, xstart, ystart, angle)
      s = Surface(*s_args)

   *Example loading from file:*

   ::

      surface = Surface('path/to/surface.irap')
      # we can also obtain the underlying pointset
      pointset = GeoPointset.fromSurface(surface)
      georegion = GeoRegion(pointset)


.. #####################################################################
.. _geo_region__select_polygon:
.. topic:: Select polygon from surface

   Will select or deselect all points from a surface contained inside a given
   polygon.


   *Example:*

   ::

      nx,ny = 12, 12
      xinc,yinc = 1, 1
      xstart,ystart = -1, -1
      angle = 0.0
      s_args = (None, nx, ny, xinc, yinc, xstart, ystart, angle)
      surface = Surface(*s_args)  # an irap surface
      pointset = GeoPointset.fromSurface(surface)
      georegion = GeoRegion(pointset)
      points = [(-0.1,2.0), (1.9,8.1), (6.1,8.1), (9.1,5), (7.1,0.9)]
      polygon = CPolyline(name='test_polygon', init_points=points)

      georegion.select_inside(polygon)
      georegion.select_outside(polygon)
      georegion.deselect_inside(polygon)
      georegion.select_polygon(polygon, inside=False, select=False)  # deselect outside


.. #####################################################################
.. _geo_region__select_halfspace:
.. topic:: Select halfspace from surface

   Will select or deselect all points from a surface above or below a line.


   *Example:*

   ::

      surface = Surface(...)  # an irap surface, see above
      pointset = GeoPointset.fromSurface(surface)
      georegion = GeoRegion(pointset)
      line = [(-0.1,2.0), (1.9,8.1)]

      georegion.select_above(line)
      georegion.deselect_above(line)
      georegion.select_below(line)
      georegion.select_halfspace(line, above=False, select=False)  # deselect below
