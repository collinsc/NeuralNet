function [ B ] = normalize( A )
nrmrt = sqrt(normSqr(A));
B = A./nrmrt;
end

