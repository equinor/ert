function plot_timestep(fignr , tstep , show_legend , logy)
  plot_mean = 1;
  data_size = max(tstep.forecast_size , tstep.analyzed_size);
  if tstep.use_true_time == 1
     x_0 = tstep.true_time * ones(data_size);
  else
     x_0 = tstep.time_step * ones(data_size);
  end
  
  if tstep.forecast_size > 0
     if tstep.analyzed_size > 0
        x_f = x_0 - 0.05;
        x_a = x_0 + 0.05;
     else
        x_f = x_0;
     end
  else
     x_a = x_0;
  end

  if tstep.forecast_size > 0
     if logy == 1
        semilogy(x_f    , tstep.forecast_data , 'bo');
     else
        plot(x_f    , tstep.forecast_data , 'bo');
     end
     hold on		
     if plot_mean == 1
        if logy==1
           semilogy(x_f(1) , mean(tstep.forecast_data), '-ro','LineWidth',2 , 'MarkerSize',8, 'MarkerEdgeColor','k', 'MarkerFaceColor','g')
        else  
           plot(x_f(1) , mean(tstep.forecast_data),'-ro','LineWidth',2 , 'MarkerSize',8, 'MarkerEdgeColor','k', 'MarkerFaceColor','g')
        end
     end
  end

  if tstep.analyzed_size > 0
     if logy == 1
        semilogy(x_a , tstep.analyzed_data , 'ro');
     else
        plot(x_a , tstep.analyzed_data , 'ro');
     end
     hold on		
     if plot_mean == 1
        if logy == 1
           semilogy(x_a(1) , mean(tstep.analyzed_data),'-ro','LineWidth',2 , 'MarkerSize',8, 'MarkerEdgeColor','k', 'MarkerFaceColor','g')
        else
           plot(x_a(1) , mean(tstep.analyzed_data),'-ro','LineWidth',2 , 'MarkerSize',8, 'MarkerEdgeColor','k', 'MarkerFaceColor','g')
        end
     end
  end

  if show_legend == 1
     if tstep.forecast_size * tstep.analyzed_size > 0
        legend('Forecast','Analyzed');
     elseif tstep.forecast_size > 0
        legend('Forecast');
     elseif tstep.analyzed_size  > 0
        legend('Analyzed');
     end
  end	

