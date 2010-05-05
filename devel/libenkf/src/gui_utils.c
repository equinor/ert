
int get_plot_data(main , node , iens , key_index, state_enum state , time_t * x_time, double * y) {
	member_config = enkf_main_iget_member_config(main, iens);
	stop_time = member_config_get_last_restart_nr(member_config)

	int count = 0;
	for(int step = 0; step <= stop_time; step++) {
		if (enkf_fs_has_node(fs, config_node, step, iens, state)) {
			double sim_days = member_config_iget_sim_days(member_config, step, fs)
			time_t sim_time = member_config_iget_sim_time(member_config, step, fs)

			enkf_fs_fread_node(fs, node, step, iens, state)
			bool valid;
			double value = enkf_node_user_get(node, key_index, &valid)


			if (valid) {
				x_time[count] = sim_time;
				x_days[count] = sim_days;
				y[count] = y;
				count++;
			}
		}
	}
}

