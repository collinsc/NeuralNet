function [W1,b1,W2,b2] = vanillaTrain(P,T,W1,b1,W2,b2,batchCount,isPlot)
Q = size(P,2);

%important constants for training
e_old = inf;
rate = 0.09;
if isPlot
    outIdx = 0;
    yError(1:(batchCount)) = 0;
end
for i = 1:batchCount
    for j = 1:Q
        [ gW1, gb1, gW2, gb2] = getGradients(W1, b1, ...
                                                        W2, b2, ... 
                                                        P(:,j),T(:,j) );
        W1 = W1 - rate*gW1;    b1 = b1 - rate*gb1;
        W2 = W2 - rate*gW2;    b2 = b2 - rate*gb2;
    end
        if isPlot
        outIdx = outIdx +1;
        e_new =  perfIndex(W1,b1,W2,b2,P,T);
        e_old = e_new;
        yError(outIdx) = e_old;
    end
end
if isPlot
    figure()
    hold on
    title('Iterations v. Performance Index')
    xlabel('Iterations')
    ylabel('Performance Index')
    numItr = 1:size(yError,2);
    plot(numItr(:,1:outIdx), yError(:,1:outIdx))
    hold off
end
end




