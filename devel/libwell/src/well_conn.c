/*
   Copyright (C) 2011  Statoil ASA, Norway. 
   
   The file 'well_conn.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <stdbool.h>

#include <util.h>

#include <ecl_kw.h>

#include <well_const.h>
#include <well_conn.h>


well_conn_type * well_conn_alloc_wellhead( const ecl_kw_type * iwel_kw , const ecl_intehead_type * header , int well_nr)  {
  const int iwel_offset = header->niwelz * well_nr;
  int conn_i = ecl_kw_iget_int( iwel_kw , iwel_offset + IWEL_HEADI_ITEM );
  
  if (conn_i > 0) {
    well_conn_type * conn = util_malloc( sizeof * conn , __func__ );
    
    conn->i    = ecl_kw_iget_int( iwel_kw , iwel_offset + IWEL_HEADI_ITEM ) - 1;
    conn->j    = ecl_kw_iget_int( iwel_kw , iwel_offset + IWEL_HEADJ_ITEM ) - 1;
    conn->k    = ecl_kw_iget_int( iwel_kw , iwel_offset + IWEL_HEADK_ITEM ) - 1;
    conn->open    = true;  // This is not really specified anywhere.
    conn->branch  = 0;
    conn->segment = 0;
    return conn;
  } else
    // The well is completed in this LGR - however the wellhead is in another LGR.
    return NULL;
}


/*
  Observe that the (ijk) and branch values are shifted to zero offset to be
  aligned with the rest of the ert libraries.  
*/
well_conn_type * well_conn_alloc( const ecl_kw_type * icon_kw , 
                                  const ecl_kw_type * iseg_kw , 
                                  const ecl_intehead_type * header , 
                                  int well_nr , 
                                  int seg_well_nr , 
                                  int conn_nr ) {
  
  const int icon_offset = header->niconz * ( header->ncwmax * well_nr + conn_nr );
  int IC = ecl_kw_iget_int( icon_kw , icon_offset + ICON_IC_ITEM );
  if (IC > 0) {
    well_conn_type * conn = util_malloc( sizeof * conn , __func__ );
  
    conn->i       = ecl_kw_iget_int( icon_kw , icon_offset + ICON_I_ITEM ) - 1;
    conn->j       = ecl_kw_iget_int( icon_kw , icon_offset + ICON_J_ITEM ) - 1;
    conn->k       = ecl_kw_iget_int( icon_kw , icon_offset + ICON_K_ITEM ) - 1;
    conn->segment = ecl_kw_iget_int( icon_kw , icon_offset + ICON_SEGMENT_ITEM ) - 1;
    {
      int int_status = ecl_kw_iget_int( icon_kw , icon_offset + ICON_STATUS_ITEM );
      if (int_status > 0)
        conn->open = true;
      else
        conn->open = false;
    }
    
    {
      int int_direction = ecl_kw_iget_int( icon_kw , icon_offset + ICON_DIRECTION_ITEM );
      if (int_direction == ICON_DEFAULT_DIR_VALUE)
        int_direction = ICON_DEFAULT_DIR_TARGET;
      
      switch (int_direction) {
      case(ICON_DIRX):
        conn->dir = well_conn_dirX;
        break;
      case(ICON_DIRY):
        conn->dir = well_conn_dirY;
        break;
      case(ICON_DIRZ):
        conn->dir = well_conn_dirZ;
        break;
      case(ICON_FRACX):
        conn->dir = well_conn_fracX;
        break;
      case(ICON_FRACY):
        conn->dir = well_conn_fracY;
        break;
      default:
        util_abort("%s: icon direction value:%d not recognized\n",__func__ , int_direction);
      }
    }

    /**
       For multisegmented wells ONLY the global part of the restart
       file has segment information, i.e. the ?SEG
       keywords. Consequently iseg_kw will be NULL for the part of a
       MSW + LGR well.
    */
    if (iseg_kw != NULL) {
      if (conn->segment >= 0) {
        const int iseg_offset = header->nisegz * ( header->nsegmx * seg_well_nr + conn->segment );
        conn->branch = ecl_kw_iget_int( iseg_kw , iseg_offset + ISEG_BRANCH_ITEM );  
      } else
        conn->branch = 0;
    } else
      conn->branch = 0;
    
    return conn;
  } else
    return NULL;  /* IC < 0: Connection not in current LGR. */
}


void well_conn_free( well_conn_type * conn) {
  free( conn );
}


void well_conn_free__( void * arg ) {
  well_conn_type * conn = (well_conn_type *) arg;
  well_conn_free( conn );
}
