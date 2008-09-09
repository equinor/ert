function ens_plot(prior_path , posterior_path , well_list , var_list , out_path , in_device)
% ens_plot(prior_path , posterior_path , well_list , var_list , {out_path , in_device})
%
% The ens_plot() function is designed to plot the results of ensemble
% experiments. Before calling ens_plot() you must use the EnKF 11-e
% option to generate ensemble plot files (say 'n' to Tecplot header).
%
%  prior_path    : Path to plot files from the prior ensemble.
% 
%  posterior_path: Path to plot files from the posterior ensemble.
% 
%  well_list     : A list of the wells you want to consider, the format 
%                  for this variable is: {'Well1', 'Well2'}.
% 
%  var_list      : A list of variables you want to consider, {'WGOR','WOPR'}.
% 
% The variables out_path and in_device are for saving to file, they are optional, see
% the documentation of diag_plot() for details.


   if exist(prior_path,'dir') == 0
      disp(sprintf('Could not locate prior_path: %s - returning from ens_plot.' , prior_path));
      return;
   end

   if posterior_path ~= 0
      if exist(posterior_path,'dir') == 0
         disp(sprintf('Could not locate posterior_path: %s - returning from ens_plot. ', posterior_path));
         return;
      end
   end

   def_device = 'png';
   lw        = 0.75;
   psize     = 4.00;
   nwell     = max(size(well_list));
   nvar      = max(size(var_list));
   sep       = filesep;

   fig_nr = 0;
   for iw=1:nwell,
       for ivar = 1:nvar,
           well = char(well_list(iw));
           var  = char(var_list(ivar));
           fig_nr = fig_nr + 1;
           
           if posterior_path ~= 0
              posterior_file = strcat(posterior_path , sep , well , '.' , var);

   	      if exist(posterior_file , 'file') == 0;
	         disp(sprintf('Could not find file: %s - returning from ens_plot',posterior_file));
   	         return;
   	      end
	      [plot_title , var , unit , time_step , posterior_dates , history , posterior] = ens_load(posterior_file);
              posterior_size  = size(posterior , 2);
           end

           prior_file     = strcat(prior_path , sep , well , '.' , var);
           if exist(prior_file , 'file') == 0;
	      disp(sprintf('Could not find file: %s - returning from ens_plot',prior_file));
   	      return;
   	   end
           [plot_title , var , unit , time_step , prior_dates , history , prior] = ens_load(prior_file);
           prior_size = size(prior , 2);
           
   
           figure(fig_nr)
	   plist = [];
           grid 
	   plot(prior_dates , history , 'ko' , 'MarkerFaceColor','k', 'MarkerSize', psize);

           hold on
           p = plot(prior_dates     , prior(:,1)     , 'b' , 'LineWidth' , lw); 
	   set(p , 'userdata' , 'Prior: 1');
	   plist = [plist , p];

           if posterior_path ~= 0 
              p = plot(posterior_dates , posterior(:,1) , 'r' , 'LineWidth' , lw);
	      set(p , 'userdata' , 'Posterior: 1');
	      plist = [plist , p];
              legend('History', 'Prior' , 'Posterior')
           else
              legend('History', 'Prior')
           end



           xlabel('Date');
           ylabel(sprintf('%s (%s)',var,unit));
	   title(sprintf('%s',plot_title));
	      
           for i=1:prior_size,
	      p = plot(prior_dates , prior(:,i) , 'b' , 'LineWidth' , lw) ; 
	      set(p , 'userdata' , sprintf('Prior: %d',i));
	      plist = [plist , p];
           end
   
           if posterior_path ~= 0
              for i=1:posterior_size,
                  p = plot(posterior_dates , posterior(:,i) , 'r' , 'LineWidth' , lw);
	          set(p , 'userdata' , sprintf('Posterior: %d',i));
	          plist = [plist , p];
              end
           end
           plot(prior_dates , history , 'ko' , 'MarkerFaceColor','k', 'MarkerSize', psize);
           hold off


	   set(gcf,'windowbuttonupfcn','h=findobj(gca,''type'',''text'');delete(h)')
	   cb='pos=get(gca,''currentpoint'');h = text(pos(1,1),pos(1,2),get(gco,''userdata'')); set(h , ''FontSize'' , 12)';
	   set(plist , 'buttondownfcn',cb);   
           datetick('x',1);
 
           if nargin >= 5,
               if out_path ~= 0,
                   out_file = sprintf('%s%s%s-%s',out_path , sep , well , var);
                   if nargin == 5,
                       device = def_device;
                   else
                       device = in_device;
                   end
                   print(fig_nr , strcat('-d',device) , out_file);
               end
           end
       end
   end
   
       
   
   
