if (active != NULL) {
  for (node_index = node_offset; node_index < node_size; node_index++) {
    if (active[node_index]) {
      int global_serial_index = serial_offset + serial_stride * serial_index;
      if (global_serial_index > serial_size)
	util_abort("%s:%d fatal error global_serial_index:%d  serial_size:%d \n",__func__ , __LINE__ , global_serial_index , serial_size);
      
      serial_data[global_serial_index] = node_data[ node_index ];
      serial_index++;
      if (serial_offset + serial_stride * serial_index >= serial_size) {
	if (node_index < (node_size - 1)) *complete = false;
	break;
      }
    }
  }
} else {
  for (node_index = node_offset; node_index < node_size; node_index++) {
    int global_serial_index = serial_offset + serial_stride * serial_index;
    if (global_serial_index > serial_size || global_serial_index < 0) 
      util_abort("%s:%d fatal error global_serial_index:%d  serial_size:%d \n",__func__ , __LINE__ , global_serial_index , serial_size);
    serial_data[global_serial_index] = node_data[ node_index ];
    serial_index++;
    if (serial_offset + serial_stride * serial_index >= serial_size) {
      if (node_index < (node_size - 1)) *complete = false;
      break;
    }
  }
}

