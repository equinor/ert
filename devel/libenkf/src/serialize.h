{
   size_t node_index;
   size_t serial_index = 0;
   serial_state->node_index2 = node_size;  /* Assuming that we get through the complete object. */ 
   serial_state->complete    = true;

   if (active != NULL) {
     for (node_index = node_index1; node_index < node_size; node_index++) {
       int global_serial_index = serial_offset + serial_stride * serial_index;
       if (active[node_index]) {
         if (global_serial_index > serial_size)
	   util_abort("%s: fatal error global_serial_index:%d  serial_size:%d \n",__func__ , global_serial_index , serial_size);
         
         serial_data[global_serial_index] = node_data[ node_index ];
         serial_index++;
         global_serial_index += serial_stride;
         elements_added++;
      }
      if (global_serial_index >= serial_size) {
	 if (node_index < (node_size - 1)) {
	   /* We did not get through the complete object after all ... */
            serial_state->complete = false;
   	    serial_state->node_index2 = node_index + 1;
         }
	 break;
       }
     }
   } else {
     for (node_index = node_index1; node_index < node_size; node_index++) {
       int global_serial_index = serial_offset + serial_stride * serial_index;
       if (global_serial_index > serial_size || global_serial_index < 0) 
         util_abort("%s: fatal error global_serial_index:%d  serial_index:%d serial_size:%d \n",__func__ , global_serial_index , serial_index , serial_size);

       serial_data[global_serial_index] = node_data[ node_index ];
       serial_index++;
       global_serial_index += serial_stride;
       elements_added++;
       if (global_serial_index >= serial_size) {
         if (node_index < (node_size - 1)) {
	    /* We did not get through the complete object after all ... */
            serial_state->complete = false;
    	    serial_state->node_index2 = node_index + 1;
         } 
         break;
       }
     }
   }   
}

