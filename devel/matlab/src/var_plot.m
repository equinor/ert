function var_plot(days , data , T , fignr , out_file , device)
  figure(fignr)
  plot(days , data(:,1) , 'o' ,  'MarkerSize',7, 'MarkerEdgeColor','k', 'MarkerFaceColor','k')
  hold on
  errorbar(days , data(:,2) , 3*data(:,3) , '-ro' , 'Color' ,'r', 'MarkerSize',5, 'MarkerEdgeColor','k', 'MarkerFaceColor','g')
  datetick('x',1);	    
  title(T)
  hold off

  if out_file ~= 0,
      print(fignr , strcat('-d',device) , out_file);
  end
