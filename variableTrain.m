
function [W1,b1,W2,b2] = variableTrain(P,T,hidden, epochs,isPlot)
%get sizes of network layers
R = size(P,1);
S1 = hidden;
S2 = size(T,1);
itrDisp = 2000;
%get randomized weights and biases
maxRand = 1;
minRand = -1;
W1 = rangedRand(minRand, maxRand, S1, R);
b1 = rangedRand(minRand, maxRand, S1, 1);
W2 = rangedRand(minRand, maxRand, S2, S1);
b2 = rangedRand(minRand, maxRand, S2, 1);
zeta = 0.03;
rho = 0.05;
eta = 2;
%important constants for training
alpha = 0.1;
maxItr=size(T,2)*epochs;

e_0 = Inf;
e = Inf;
itr = 0;
numItr = 1:maxItr;
yErr(1:maxItr) = 0;
yRate(1:maxItr) = 0;
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
        e_0 = (t-a2);
        %back propigation
        s2 = -2 * diag((ones(size(a2)) - a2).*a2) * e_0;
        s1 =diag((ones(size(a1)) - a1).*a1)*(W2')*s2; 
        %weight update
        W2_n = W2 - alpha * s2*(a1');
        b2_n = b2 - alpha * s2;
        W1_n = W1 - alpha * s1*(a0');
        b1_n = b1 - alpha * s1;
        e = (t - logsigmoid(W2_n*logsigmoid(W1_n*a0 + b1_n) + b2_n));
        if index(e) > index(e_0)
            if(index(e) - index(e_0))/(e'*e)*100 < zeta
                W2 = W2_n;
                b2 = b2_n;
                W1 = W1_n;
                b1 = b1_n;
            else
                alpha = alpha * rho;
            end
        else
            alpha = alpha * eta;
            W2 = W2_n;
            b2 = b2_n;
            W1 = W1_n;
            b1 = b1_n;
        end
        
        
        itr = itr +1;
        if mod(  itr, itrDisp) == 0
           fprintf('        epoch: %i, \titr: %i,\t err: %f\n', i, itr,e) 
        end
        yErr(itr) = index(e);
        yRate(itr) = alpha;
    end
    fprintf('    epoch %i completed\n',i)
end
fprintf('    Network trained, iterations: %i, final error: %f\n', itr, e);
if isPlot == true
    figure()
    hold on
    title('Iterations v. Mean Squared Error')
    xlabel('Iterations')
    ylabel('Sum of Squared Error')
    plot(numItr, yErr, numItr, yRate)
    hold off
end
end

function idx = index(x)
    idx = x'*x;
end

function out = rangedRand(min, max, R, C)
out = min + (max - min) * rand(R,C);
end