function [title ,  var , unit ,time_step , true_time ,  history , ens_data] = ens_load(filename)
fid       = fopen(filename , 'rb');

title     = fread_string(fid);
var  	  = fread_string(fid);
unit 	  = fread_string(fid);
ens_size  = fread(fid , 1 , 'int');
time_size = fread(fid , 1 , 'int');

buffer = reshape(fread(fid , time_size * 3 , 'int') , 3 , time_size);
days   = buffer(1,:);
months = buffer(2,:);
years  = buffer(3,:);
clear buffer;

true_time = datenum(years , months , days);
history   = fread(fid , time_size , 'double');
time_step = 0:1:time_size - 1;
buffer    = fread(fid , time_size * ens_size , 'double');
ens_data  = reshape(buffer , ens_size , time_size)';

fclose(fid);
