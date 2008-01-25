function [data , plot_title]  = fread_thist(filename,show_mean)
  fid   = fopen(filename,'rb');
  plot_title = fread_string(fid);
  time_length = fread(fid , 1 , 'int');

  data = [];
  for itime = 1:time_length
     [active , tstep] = fread_timestep(fid);
     if active == 1
        data = [data; tstep];
         if show_mean == 1
            disp(sprintf('Analyzed mean: %6.4f +/- %6.4f ',mean(tstep.analyzed_data) , std(tstep.analyzed_data)));
         end
     end
  end
  fclose(fid);
