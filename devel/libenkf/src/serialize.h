{
   double * serial_data = serial_vector->serial_data;
   int    serial_stride = serial_vector->serial_stride;
   size_t serial_size   = serial_vector->serial_size;
 
   for (active_index = current_active_node_index; active_index < active_node_offset + active_size; active_index++) {
     int global_serial_index = serial_offset + serial_stride * serial_state->serial_index;
     int node_index;
     if (global_serial_index > serial_size)
       util_abort("%s: fatal error global_serial_index:%d  serial_size:%d \n",__func__ , global_serial_index , serial_size);

     if (active_size < node_size)
       node_index = active_list[active_index - active_node_offset];  /* Not all elements are active */
     else
       node_index = active_index - active_node_offset;               /* All elements are active: active_list is not accessed. */

     serial_data[global_serial_index] = node_data[ node_index ];
     serial_state->serial_index++;
     global_serial_index += serial_stride;
     elements_added++;
     
     if (global_serial_index >= serial_size) {
     	 break;
     }
   } 
}

