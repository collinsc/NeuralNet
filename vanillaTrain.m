function [W1,b1,W2,b2, index] = vanillaTrain(P,T,W1,b1,W2,b2,isPlot)
Q = size(P,2);
%important constants for training
rate = 0.09;
for j = 1:Q
    [ gW1, gb1, gW2, gb2] = getGradients(W1, b1, ...
                                                    W2, b2, ... 
                                                    P(:,j),T(:,j) );
    W1 = W1 - rate*gW1;    b1 = b1 - rate*gb1;
    W2 = W2 - rate*gW2;    b2 = b2 - rate*gb2;
end
index=  perfIndex(W1,b1,W2,b2,P,T);
if isPlot
    fprintf('\tvanilla performance index: %f\n', index)
end
end




