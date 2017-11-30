function [W1,b1,W2,b2] = conjugateTrain(P,T,W1,b1,W2,b2, batchCount,isPlot)
%count of elements in training set
Q = size(P,2);
%performance preallocation
gW1 = zeros(size(W1));    gb1 = zeros(size(b1));
gW2 = zeros(size(W2));    gb2 = zeros(size(b2));
gW1n = zeros(size(W1));   gb1n = zeros(size(b1));
gW2n = zeros(size(W2));   gb2n = zeros(size(b2));
%important constants for training
epsilon = 0.5;         %rate to increase search interval
startingRate =  0;     %minimum jump
%reset search direction every n iterations
R = size(P,1);
s1 = size(W1,1);
s2 = size(W2,1);
resetPeriod = R*s2 + s2 + s2*s1 + s1;  
e_old = inf;
if isPlot   %data visualization only
    outIdx = 0;
    yError(1:(batchCount)) = 0;
    yRate(1:(batchCount)) = 0;
end

%initialize gradients for first iteration
for j = 1:Q
    [gW1t, gb1t, gW2t, gb2t] = getGradients(    W1, b1, ...
                                                W2, b2, ... 
                                                P(:,j),T(:,j));
    %accumulate an average gradient
    gW1 = gW1 + gW1t/Q;     gb1 = gb1 + gb1t/Q;
    gW2 = gW2 + gW2t/Q;     gb2 = gb2 + gb2t/Q;
end
%initialize search directions with normalized gradients
pW1 = -normalize(gW1);    pb1 = -normalize(gb1); 
pW2 = -normalize(gW2);    pb2 = -normalize(gb2);
for i = 2:batchCount
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
    if isPlot
        fprintf('\titeration: %i,    performance index: %f,    learning rate: %f\n', outIdx, e_old, rate)
        outIdx = outIdx + 1; 
        yError(outIdx) = e_old;
        yRate(outIdx) = rate;
    end
    %get new gradients
    for j = 1:Q
        [gW1nt, gb1nt, gW2nt, gb2nt] = getGradients(        W1, b1, ...
                                                            W2, b2, ... 
                                                            P(:,j),T(:,j));                                           

        gW1n = gW1n + gW1nt/Q;     gb1n = gb1n + gb1nt/Q;
        gW2n = gW2n + gW2nt/Q;     gb2n = gb2n + gb2nt/Q;
    end
    %get directional gain
    if mod(i,resetPeriod) == 0
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
    pW1 = normalize(pW1n);          pb1 = normalize(pb1n);
    pW2 = normalize(pW2n);          pb2 = normalize(pb2n);
    gW1 = gW1n;          gb1 = gb1n;
    gW2 = gW2n;          gb2 = gb2n;

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
    figure()
    hold on
    title('Iterations v. LearningRate')
    xlabel('Iterations')
    ylabel('LearningRate')
    numItr = 1:size(yError,2);
    plot(numItr(:,1:outIdx), yRate(:,1:outIdx))
    hold off
end
end




