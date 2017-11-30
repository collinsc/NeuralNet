
function [W1,b1,W2,b2] = conjugateTrainTest(P,T,hidden, epochs,isPlot)
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
epsilon = 1;
startingRate = 0.001;
resetPeriod = R*S2 + S2 + S2*S1 + S1;
itr = 0;
sError(1:epochs) = 0;
p_W2 = zeros(size(W2));
p_b2 = zeros(size(b2));
p_W1 = zeros(size(W1));
p_b1 = zeros(size(b1));


for i = 1:epochs
    [g_W1_0, g_b1_0, g_W2_0, g_b2_0] = getGrad(W1, b1, W2, b2, P, T);
    if mod(itr, resetPeriod) == 0
        p_W1 = -g_W1_0;
        p_b1 = -g_b1_0;
        p_W2 = -g_W2_0;
        p_b2 = -g_b2_0;
    end  
    [a, b] = getInterval(@(rate) perfIndex(    W1 - rate*p_W1, ...
                    b1 + rate*p_b1, ...
                    W2 + rate*p_W2, ...
                    b2 + rate*p_b2, ...
                    P,T), startingRate, epsilon);
    rate = golden(@(rate) perfIndex(    W1 - rate*p_W1, ...
                    b1 + rate*p_b1, ...
                    W2 + rate*p_W2, ...
                    b2 + rate*p_b2, ...
                    P,T), a, b);
    itr = itr + 1;
    e_0 = perfIndex(W1,b1, W2, b2, P,T);  
    sError(itr) = e_0;
    W1 = W1 + rate*p_W1;
    b1 = b1 + rate*p_b1;
    W2 = W2 + rate*p_W2';
    b2 = b2 + rate*p_b2';
    %todo delete
    e = perfIndex(W1,b1, W2, b2, P,T);
    [g_W1, g_b1, g_W2, g_b2] = getGrad(W1, b1, W2, b2, P, T);
    getBeta = @(g, g_0) (g'*g)/(g_0'*g_0);
    b_W1 = getBeta(g_W1, g_W1_0);
    b_b1 = getBeta(g_b1, g_b1_0);
    b_W2 = getBeta(g_W2, g_W2_0);
    b_b2 = getBeta(g_b2, g_b2_0);
    p_W1 = -g_W1 + b_W1.*p_W1;
    p_b1 = -g_b1 + b_b1.*p_b1;
    p_W2 = -g_W2 + b_W2.*p_W2;
    p_b2 = -g_b2 + b_b2.*p_b2;
    fprintf('    epoch %i completed\n',i)
end
fprintf('    Network trained, iterations: %i\n', itr);
if isPlot == true
    figure()
    hold on
    title('Iterations v. Performance Index')
    xlabel('Iterations')
    ylabel('Performance Index')
    numItr = 1:size(sError,2);
    plot(numItr, sError)
    hold off
end
end

function [  g_W2, g_b2, g_W1, g_b1 ] = getGrad(W1,b1, W2,b2, P, T) 
    g_W2 = 0; g_b2 = 0; g_W1= 0; g_b1 = 0;
    for i = 1:size(T,2)
        %forward propigation
        a0 = P(:,i);
        t = T(:,i);
        %get the first layer output
        a1 = logsigmoid(W1*a0 + b1);
        %get the second layer output
        a2 = purelin(W2*a1 + b2);
        %error
        e = (t - a2);
        %back propigation
        s2 = -2 .* diag(ones(size(a2))) * e;
        s1 = diag((ones(size(a1)) - a1).*a1)*(W2')*s2;
        aveSum = @(x,q) sum(x./q,2);
        %accumulate gradients       
        g_W2 = g_W2 + aveSum(s2*(a1'), size(T,2));
        g_b2 = g_b2 + aveSum(s2, size(T,2));
        g_W1 = g_W1 + aveSum(s1*(a0'),size(T,2));
        g_b1 = g_b1 + aveSum(s1, size(T,2));
    end
end


function index = perfIndex(W1, b1, W2, b2, a0, T)
a2 = purelin(W2*logsigmoid(W1*a0 + b1) + b2);
e = (T - a2)';
index = sum(e'*e)/size(T,2);
end



function out = rangedRand(min, max, R, C)
out = min + (max - min) * rand(R,C);
end