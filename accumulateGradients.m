function [ gW1, gb1, gW2, gb2] = accumulateGradients( W1, b1, W2, b2, P, T )
gW1 = zeros(size(W1,1),size(P,1));    gb1 = zeros(size(b1,1),1);
gW2 = zeros(size(W2,1),size(W1,1));    gb2 = zeros(size(b2,1),1);
%count of elements in training set
Q = size(P,2);
%initialize gradients for first iteration
for j = 1:Q
    [gW1t, gb1t, gW2t, gb2t] = getGradients(    W1, b1, ...
                                                W2, b2, ... 
                                                P(:,j),T(:,j));
    %accumulate an average gradient
    gW1 = gW1 + gW1t/Q;     gb1 = gb1 + gb1t/Q;
    gW2 = gW2 + gW2t/Q;     gb2 = gb2 + gb2t/Q;
end
end


