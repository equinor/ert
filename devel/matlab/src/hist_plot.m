function hist_plot(tstep)
  w = 0.75;
  if tstep.forecast_size > 0
     [f  , xf] = hist(tstep.forecast_data , sqrt(tstep.forecast_size));
     bar(xf,f,w,'b');
     hold on
  end
  if tstep.analyzed_size > 0
     [a  , xa] = hist(tstep.analyzed_data , sqrt(tstep.analyzed_size));
     bar(xa,a,w,'r');
  end
  hold off
