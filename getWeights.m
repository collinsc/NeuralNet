function [W1,b1,W2,b2,W3,b3] = getWeights(P,T, h1, h2)
R = size(P,1);
S1 = h1;
S2 = h2;
S3 = size(T,1);
%get randomized weights and biases
maxRand = 0.5;
minRand = -0.5;
W1 = rangedRand(minRand, maxRand, S1, R);
b1 = rangedRand(minRand, maxRand, S1, 1);
W2 = rangedRand(minRand, maxRand, S2, S1);
b2 = rangedRand(minRand, maxRand, S2, 1);
W3 = rangedRand(minRand, maxRand, S3, S2);
b3 = rangedRand(minRand, maxRand, S3, 1);
end

