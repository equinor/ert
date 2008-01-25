struct enkf_config_struct {
  int  		    ens_size;
  hash_type        *config_hash;
  hash_type        *obs_hash;
  bool              endian_swap;
  path_fmt_type    * run_path;
  int                Nwells;
  char            **well_list;
};
