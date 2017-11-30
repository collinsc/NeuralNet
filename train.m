
function [W1,b1,W2,b2] = train(P,T,hidden, epochs,isPlot)
%get sizes of network layers
R = size(P,1);
S1 = hidden;
S2 = size(T,1);
%get randomized weights and biases
maxRand = 1;
minRand = -1;
W1 = rangedRand(minRand, maxRand, S1, R);
b1 = rangedRand(minRand, maxRand, S1, 1);
W2 = rangedRand(minRand, maxRand, S2, S1);
b2 = rangedRand(minRand, maxRand, S2, 1);
%important constants for training
alpha = 0.1;
maxItr=size(T,2)*epochs;

e = Inf;
itr = 0;
numItr = 1:maxItr;
sError(1:maxItr) = 0;

for i = 1:epochs
    epoch = randperm(size(T,2));
    for j = 1:size(T,2)
        %forward propigation
        a0 = P(:,epoch(j));
        t = T(:,epoch(j));
        %get the first layer output
        a1 = logsigmoid(W1*a0 + b1);
        %get the second layer output
        a2 = logsigmoid(W2*a1 + b2);
        %error
        e = t - a2;
        %back propigation
        s2 = -2 * diag((ones(size(a2)) - a2).*a2) * e;
        s1 =diag((ones(size(a1)) - a1).*a1)*(W2')*s2;
        %weight update
        W2 = W2 - alpha * s2*(a1');
        b2 = b2 - alpha * s2;
        W1 = W1 - alpha * s1*(a0');
        b1 = b1 - alpha * s1;
        itr = itr +1;
        sError(itr) = sum(e'*e);
    end
    fprintf('    epoch %i completed\n',i)
end
fprintf('    Network trained, iterations: %i, max: %i\n', itr, itr);
if isPlot == true
    figure()
    hold on
    title('Iterations v. Mean Squared Error')
    xlabel('Iterations')
    ylabel('Sum of Squared Error')
    plot(numItr, sError)
    hold off
end
end

function out = rangedRand(min, max, R, C)
out = min + (max - min) * rand(R,C);
end