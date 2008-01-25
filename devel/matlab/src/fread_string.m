function [s] = fread_string(fid)
  len = fread(fid , 1 , 'int');
  s   = fread(fid , len , 'uint8=>char')';
  fread(fid , 1 , 'uint8=>char'); % Skipping the trailing \0
