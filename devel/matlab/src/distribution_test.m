% One sample Kolmogorov-Smirnov test
% This function performs a Kolmogorov-Smirnov test to compare the values in
% the data vector d with a standard normal distribution N(0,1)
%The result H is 1 if you can reject the hypothesis that X has a standard normal distribution, 
%or 0 if you cannot reject that hypothesis. You reject the hypothesis if the test is significant at 
%the alpha percent level.
% H = 0 accept 
% H = 1 reject
% H = -1 no data available
function [H_a,H_f]=distribution_test(d,alpha)
H_f=[];
H_a=[];
N=size(d,1);
for i=1:N
  if d(i).forecast_size > 0 
    Z=(d(i).forecast_data-mean(d(i).forecast_data))/std(d(i).forecast_data);
    H_f(i)=kstest(Z,[],alpha,0);
  else
    H_f(i)=-1;
  end
  if d(i).analyzed_size > 0 
    Z=(d(i).analyzed_data-mean(d(i).analyzed_data))/std(d(i).analyzed_data);
    H_a(i)=kstest(Z,[],alpha,0);
  else
    H_a(i)=-1;
  end
end
