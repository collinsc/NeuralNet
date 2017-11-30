function [W1,b1,W2,b2] = conjugateCompareTrain(P,T,hidden1, epochs,isPlot)
%get sizes of network layers
R = size(P,1);
S1 = hidden1;
S2 = hidden2;
S3 = size(T,1);
Q = size(P,2);
%get randomized weights and biases
maxRand = 0.5;
minRand = -0.5;
W1 = rangedRand(minRand, maxRand, S1, R);
b1 = rangedRand(minRand, maxRand, S1, 1);
W2 = rangedRand(minRand, maxRand, S2, S1);
b2 = rangedRand(minRand, maxRand, S2, 1);
W3 = rangedRand(minRand, maxRand, S3, S2);
b3 = rangedRand(minRand, maxRand, S3, 1);
%important constants for training
e_old = inf;
rate = 0.09;
itr = 0;
yError(1:(epochs)) = 0;
yRate(1:(epochs)) = 0;
for i = 1:epochs
    for j = 1:Q
        [ gW1, gb1, gW2, gb2, gW3, gb3 ] = getGradient( W1, b1, ...
                                                        W2, b2, ... 
                                                        W3, b3, ...
                                                        P(:,j),T(:,j) );
        %accumulate normalized gradients 
        W1 = W1 - rate*gW1;
        b1 = b1 - rate*gb1;
        W2 = W2 - rate*gW2;
        b2 = b2 - rate*gb2;
        W3 = W3 - rate*gW3;
        b3 = b3 - rate*gb3;
    end
    itr = itr +1;
    e_new =  perfIndex(W1,b1,W2,b2,W3,b3,P,T);
    if e_new > e_old*1.5
        break
    end
    e_old = e_new;
    yError(itr) = e_old;
    fprintf('    epoch %i completed, error: %f, rate: %f\n',i,yError(itr),yRate(itr))
end
fprintf('    Network trained, iterations: %i\n', itr);
if isPlot == true
    figure()
    hold on
    title('Iterations v. Performance Index')
    xlabel('Iterations')
    ylabel('Performance Index')
    numItr = 1:size(yError,2);
    plot(numItr(:,1:itr), yError(:,1:itr))
    hold off
end
end




