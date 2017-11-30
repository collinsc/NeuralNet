function [ T_n ] = formatOutput( T_o )
T_n = zeros(10,size(T_o,1));
lookup= diag(ones(1,10));
getCode = @(x) lookup(x,:);
for i = 1:size(T_o,1)
   T_n(:,i) = getCode(T_o(i)+1);
end
end

