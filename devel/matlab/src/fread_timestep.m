function [active,tstep] = fread_timestep(fid)
  tstep  = struct();

  active = fread(fid , 1 , 'int');
  if active == 1
     time_step 	   = fread(fid , 1 , 'int');
     tstep.time_step = time_step;

     true_time 	   = fread(fid , 1 , 'int');
     tstep.true_time =  true_time;

     use_true_time = fread(fid , 1 , 'int');
     tstep.use_true_time = use_true_time;          

     forecast_size = fread(fid , 1 , 'int');   
     tstep.forecast_size = forecast_size;
     if forecast_size > 0
        forecast_data = fread(fid , forecast_size , 'double');
        tstep.forecast_data = forecast_data;
     end

     analyzed_size = fread(fid , 1 , 'int');   
     tstep.analyzed_size = analyzed_size;
     if analyzed_size > 0
        analyzed_data = fread(fid , analyzed_size , 'double');
        tstep.analyzed_data = analyzed_data;
     end
     if tstep.analyzed_size + tstep.forecast_size == 0
        active = 0;
     end
  end  





