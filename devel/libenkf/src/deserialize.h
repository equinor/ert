{
   double * serial_data = serial_vector->serial_data;
   int    serial_stride = serial_vector->serial_stride;
   size_t active_index;

   for (active_index = current_active_node_index; active_index < active_node_index2; active_index++) {
     int global_serial_index = serial_state->serial_index * serial_stride + serial_offset;
     int node_index;

     if (active_size < node_size)
       node_index = active_list[active_index - active_node_offset];  /* Not all elements are active */
     else
       node_index = active_index - active_node_offset;               /* All elements are active: active_list is not accessed. */

     node_data[node_index] = serial_data[global_serial_index];
     serial_state->serial_index++;
   }  
}    
  
