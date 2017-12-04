function [ W1, b1,  W2,  b2, ...
           pW1,pb1, pW2, pb2, ...
           gW1, gb1, gW2, gb2, ...
           iteration] = ...
                        conjugateTrain( P, T, ...
                                        W1, b1, W2, b2, ...
                                        pW1, pb1, pW2, pb2, ...
                                        gW1,gb1, gW2, gb2, ...
                                        iteration, batchCount, isGraph)
%important constants for training

epsilon = 0.001;         %rate to increase search interval
startingRate =  0.0;     %minimum jump
%reset search direction every n iterations
R = size(P,1);
s1 = size(W1,1);
s2 = size(W2,1);
resetPeriod = R*s2 + s2 + s2*s1 + s1;  
e_old = inf;
if isGraph   
    outIdx = 0;
    yError(1:(batchCount)) = 0;
    yRate(1:(batchCount)) = 0;
end
%initialize search directions with normalized gradients
for i = 1:batchCount
    %select an interval to minimize
    [a, b] = getInterval( @(rate) perfIndex(    W1 + rate*pW1, ...
                                    b1 + rate*pb1, ...
                                    W2 + rate*pW2, ...
                                    b2 + rate*pb2, ...
                                    P,T), ...
                                    startingRate, epsilon);
    %minizize the interval to tolerance
    rate = golden(@(rate) perfIndex(    W1 + rate*pW1, ...
                                    b1 + rate*pb1, ...
                                    W2 + rate*pW2, ...
                                    b2 + rate*pb2, ...
                                    P,T),...
                                    a, b);
    e_new = perfIndex(  W1 + rate*pW1, ... 
                        b1 + rate*pb1, ... 
                        W2 + rate*pW2, ... 
                        b2 + rate*pb2, ... 
                        P, T);
    e_old = e_new;

    %conjugate gradient weight update
    W1 = W1 + rate*pW1;     b1 = b1 + rate*pb1;
    W2 = W2 + rate*pW2;     b2 = b2 + rate*pb2;
    if isGraph
        fprintf('\titeration: %i,    performance index: %f,    learning rate: %f\n', outIdx, e_old, rate)
        outIdx = outIdx + 1; 
        yError(outIdx) = e_old;
        yRate(outIdx) = rate;
    end
    [gW1n, gb1n, gW2n, gb2n] = accumulateGradients( W1, b1, W2, b2, P, T );
    iteration = iteration +1;
    %get directional gain
    if mod(iteration, resetPeriod) == 0
        bW1 = 0;    bb1 = 0;
        bW2 = 0;    bb2 = 0;
    else
        %decide how much of the new gradient to "mix" based on magnitude
        %uses fletcher reeves update
        bW1 = getGains(gW1n, gW1);    bb1 = getGains(gb1n, gb1);
        bW2 = getGains(gW2n, gW2);    bb2 = getGains(gb2n, gb2);
    end
    %calculate new directions
    pW1n = -gW1n + pW1*bW1;    pb1n = -gb1n + pb1*bb1;
    pW2n = -gW2n + pW2*bW2;    pb2n = -gb2n + pb2*bb2;
    %transfer over old values
    pW1 = pW1n;          pb1 = pb1n;
    pW2 = pW2n;          pb2 = pb2n;
    gW1 = gW1n;          gb1 = gb1n;
    gW2 = gW2n;          gb2 = gb2n;

end
if isGraph
%    figure()
%     hold on
%     title('Iterations v. Performance Index')
%     xlabel('Iterations')
%     ylabel('Performance Index')
%     numItr = 1:size(yError,2);
%     plot(numItr(:,1:outIdx), yError(:,1:outIdx))
%     hold off
%     figure()
%     hold on
%     title('Iterations v. LearningRate')
%     xlabel('Iterations')
%     ylabel('LearningRate')
%     numItr = 1:size(yError,2);
%     plot(numItr(:,1:outIdx), yRate(:,1:outIdx))
%     hold off
end
end




