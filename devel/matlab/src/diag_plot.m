function diag_plot(year , month , day , path , VAR , NList , out_path, in_device)
%diag_plot(year , month , day , path , VAR , NList , out_path , device)
%
%The diag_plot() function can be used to plot time-series of scalar 
%enkf variables like fault multipliers and the WOC. 
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
%   [VAR]: This is the name of the variable you want to plot, it is
%          typically 'multz', 'WOC' or 'multflt'.
%
%   [NList]: The scalar parameters are typically 'vectors' - i.e. there 
%            are for instance many fault multipliers. The NList 
%            argument is to designate which (e.g.) fault multipliers
%            to plot. For instance to fault multpliers 1 and 3 use 
%            [1,3] as NList argument.
%
%   The two last arguments are for generating hard copies:
%
%   [out_path]: Path were hard copies are stored.
%
%   [device]: Type of files to plot, e.g. 'png', 'eps',..., defaults
%             to 'png'.



    def_device = 'png';
    sep = filesep;
    aveFile = strcat(path , sep , VAR , 'aveA.dat');
    stdFile = strcat(path , sep , VAR , 'stdA.dat');
    ave = load(aveFile);
    std = load(stdFile);
    days = ave(:,3);
    
    
    for i = 1:max(size(NList)),
        figure(i)
        N = NList(i);
        errorbar(days , ave(:,N+3) , std(:,N + 3) , '-ro','LineWidth',2 , 'MarkerSize',10, 'MarkerEdgeColor','k', 'MarkerFaceColor','g')
        xlabel('Time (days)')
        title(sprintf('%s %d' , VAR , N));
        if nargin >= 7,
            if out_path == 1,
                out_path = path;
            end
            if out_path ~= 0,
                out_file = sprintf('%s%s%s-%d',out_path , sep , VAR , N);
                if nargin == 7,
                    device = def_device;
                else
                    device = in_device;
                end
                print(i , strcat('-d',device) , out_file);
            end
        end
    end
end
        
    
