function [data] = ins_plot(filename,logy)
% function d = ins_plot(filename)
% 
% This function is made to plot time histograms of quantities made with 
% the 11i option in the EnKF system. The function first loads the data 
% from the file 'filename', and then plots it. Observe that file 'filename' 
% is a binary file, with the following format:
% 
% 
% The ins_plot function returns the plotted data, which can be used in the
% hist_plot() function, or for arbitrary analysis. The return value is a 
% vector of structs with the following fields:
% 
% time_step(int): This is the time step this data comes from.
% 
% true_time(int): This is the C-library internal time representation of the
%     observation time. Can be used as input to the matlab date/time functions.
%     Use of this field is currently not implemented.
% 
% use_true_time(int): An integer switch 0/1 whether the true_time field is
%     set with a meaningfull value.
% 
% forecast_size(int): The length of the forecast vector.
% 
% forecast_data(double): The forceast data - observe that if forecast_size == 0, 
%     then the forecast_data field does not exist.
% 
% analyzed_size / analyzed_data: as Same forecast_size/forecast_data for analyzed
%     results.              

  [data , plot_title] = fread_thist(filename , 1);
  ins_plot__(data , plot_title , 1 , logy);
  print('-dpng', strcat(filename , '.png'));
