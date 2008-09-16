if (active != NULL) {
  for (node_index = node_offset; node_index < last_node_index; node_index++) {
    if (active[node_index]) {
      node_data[node_index] = serial_data[serial_index * serial_stride + serial_offset];
      serial_index++;
    }
  }
} else {
  for (node_index = node_offset; node_index < last_node_index; node_index++) {
    node_data[node_index] = serial_data[serial_index * serial_stride + serial_offset];
    serial_index++;
  }
}
    
  
