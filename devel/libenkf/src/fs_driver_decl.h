struct fs_driver_struct {
  load_node_ftype    * load;
  save_node_ftype    * save;
  swapout_node_ftype * swapout;
  swapin_node_ftype  * swapin;
  path_fmt_type      * path;
};
