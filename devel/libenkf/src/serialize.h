{
   double * serial_data = serial_vector->serial_data;
   int    serial_stride = serial_vector->serial_stride;
   size_t serial_size   = serial_vector->serial_size;
   

   if (active != NULL) {
     for (node_index = current_node_index; node_index < node_offset + node_size; node_index++) {
       int global_serial_index = serial_offset + serial_stride * serial_state->serial_index;
       if (active[node_index - node_offset]) {
         if (global_serial_index > serial_size)
	   util_abort("%s: fatal error global_serial_index:%d  serial_size:%d \n",__func__ , global_serial_index , serial_size);
 
         serial_data[global_serial_index] = node_data[ node_index - node_offset];
         serial_state->serial_index++;
         global_serial_index += serial_stride;
         elements_added++;
      }
      if (global_serial_index >= serial_size) {
	 break;
       }
     }
   } else {
     for (node_index = current_node_index; node_index < node_offset + node_size; node_index++) {
       int global_serial_index = serial_offset + serial_stride * serial_state->serial_index;
       if (global_serial_index > serial_size || global_serial_index < 0) 
         util_abort("%s: fatal error global_serial_index:%d  serial_index:%d serial_size:%d \n",__func__ , global_serial_index , serial_state->serial_index , serial_size);

       serial_data[global_serial_index] = node_data[ node_index - node_offset];
       serial_state->serial_index++;
       global_serial_index += serial_stride;
       elements_added++;
       if (global_serial_index >= serial_size) {
         break;
       }
     }
   }   
}

