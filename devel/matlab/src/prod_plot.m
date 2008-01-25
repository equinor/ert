function prod_plot(year,month,day , path , well , out_path, in_device)
%prod_plot(year , month , day , path , VAR , NList , out_path , device)
%
%The prod_plot() function can be used to plot time-series of production.
%
% o Prior to calling diag_plot() yous *must* use the EnKF program
%   to extract the mean and standard deviation of the ensemble. 
%   This is done by using menu option 11-p and then analyse the
%   ensaveA and ensstdA files. 
%
%   The diag_plot() function currently *only* considers the analyzed
%   files.
%
% o The function has eight arguments, only the six first are
%   mandatory:
%
%   year / month / day : The starting date of the simulation.
%
%   [path:] This is the path to where the diagnostica variable from
%           EnKF are stored. If you have run EnKF in the directory
%  
%                /path/to/EnKF_model/A
%            
%           you should enter '/path/to/EnKF_model/A/Diag' as first 
%           argument.
%
%   [well:] The name of the well
%
%   The two last arguments are for generating hard copies:
%
%   [out_path]: Path were hard copies are stored.
%
%   [device]: Type of files to plot, e.g. 'png', 'eps',..., defaults
%             to 'png'.

	      
    def_device = 'png';
    sep = filesep;
    dataFile = strcat(path , sep , well);
    data = load(dataFile);
    start_day = datenum(year,month,day);
    days = data(:,2) + start_day;
    OPR = data(:,3:5);
    GOR = data(:,6:8);
    WCT = data(:,9:11);
    if nargin == 5,
       out_path = 0;
    end
    device = def_device;
    if nargin == 7,
       device = in_device;
    end

    if out_path == 0,
       out_GOR = 0;
       out_OPR = 0;
       out_WCT = 0;
    else
       out_GOR = sprintf('%s%s%s-GOR' , out_path , sep , well);
       out_OPR = sprintf('%s%s%s-OPR' , out_path , sep , well);
       out_WCT = sprintf('%s%s%s-WCT' , out_path , sep , well);
    end     

    var_plot(days , OPR , sprintf('%s - OPR' , well) , 1 , out_OPR , device);
    var_plot(days , GOR , sprintf('%s - GOR' , well) , 2 , out_GOR , device);
    var_plot(days , WCT , sprintf('%s - WCT' , well) , 3 , out_WCT , device);
                
        
    
