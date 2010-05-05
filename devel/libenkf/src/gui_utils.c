
int get_plot_data(main, fs, config_node, node, member, key_index, x_time, x_days, y) {
	member_config = enkf_main_iget_member_config(main, member);
	stop_time = member_config_get_last_restart_nr(member_config)

	FORECAST = 2
	
	int count = 0;
	for(int step = 0; step <= stop_time; step++) {
		if (enkf_fs_has_node(fs, config_node, step, member, FORECAST)) {
			double sim_days = member_config_iget_sim_days(member_config, step, fs)
			time_t sim_time = member_config_iget_sim_time(member_config, step, fs)

			enkf_fs_fread_node(fs, node, step, member, FORECAST)
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

