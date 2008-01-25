function [data] = ins_plot__(data , plot_title , fig_nr , logy)
  figure(fig_nr)
  show_legend = 1;
  for i = 1:max(size(data))
     if i > 1
        show_legend = 0;
     end
     plot_timestep(1 , data(i) , show_legend , logy);
  end 
  title(plot_title);
  xlabel('Time');
  hold off
