{
   double * serial_data = serial_vector->serial_data;
   int    serial_stride = serial_vector->serial_stride;
   size_t serial_index = 0;
   size_t node_index;

   if (active != NULL) {
     for (node_index = node_index1; node_index < node_index2; node_index++) {
       if (active[node_index]) {
         node_data[node_index] = serial_data[serial_index * serial_stride + serial_offset];
         serial_index++;
       }
     }
   } else {
     for (node_index = node_index1; node_index < node_index2; node_index++) {
       node_data[node_index] = serial_data[serial_index * serial_stride + serial_offset];
       serial_index++;
     }
   }  
}    
  
