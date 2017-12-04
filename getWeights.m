function [W1,b1,W2,b2] = getWeights(P,T, h1)
R = size(P,1);
S1 = h1;
S2 = size(T,1);
%get randomized weights and biases
maxMag = 0.5;
minMag = 0.2;

W1 = rangedRand(minMag, maxMag, S1, R);
b1 = rangedRand(minMag, maxMag, S1, 1);
W2 = rangedRand(minMag, maxMag, S2, S1);
b2 = rangedRand(minMag, maxMag, S2, 1);
end

