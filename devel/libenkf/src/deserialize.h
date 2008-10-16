{
   double * serial_data = serial_vector->serial_data;
   int    serial_stride = serial_vector->serial_stride;
   size_t node_index;

   if (active != NULL) {
     for (node_index = current_node_index; node_index < node_index2; node_index++) {
       if (active[node_index - node_offset]) {
         node_data[node_index - node_offset] = serial_data[serial_state->serial_index * serial_stride + serial_offset];
         serial_state->serial_index++;
       }
     }
   } else {
     for (node_index = current_node_index; node_index < node_index2; node_index++) {
       node_data[node_index - node_offset] = serial_data[serial_state->serial_index * serial_stride + serial_offset];
       serial_state->serial_index++;
     }
   }  
}    
  
