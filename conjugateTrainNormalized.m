
function [W1,b1,W2,b2] = conjugateTrainNormalized(P,T,hidden, epochs,isPlot)
%get sizes of network layers
R = size(P,1);
S1 = hidden;
S2 = size(T,1);
%get randomized weights and biases
maxRand = 10;
minRand = -10;
W1 = rangedRand(minRand, maxRand, S1, R);
b1 = rangedRand(minRand, maxRand, S1, 1);
W2 = rangedRand(minRand, maxRand, S2, S1);
b2 = rangedRand(minRand, maxRand, S2, 1);
%important constants for training
epsilon = 0.5;
startingRate = 0.000;
resetPeriod = R*S2 + S2 + S2*S1 + S1;
itr = 1;
sError(1:epochs) = 0;
%initial training epoch
%get the first layer output
A1 = logsigmoid(W1*P + b1);
%get the second layer output
A2 = logsigmoid(W2*A1 + b2);
%error
E = T - A2;
%back propigation
S2 = -2.*E;
S1 = A1.*(1 - A1).*(W2'*S2);
sError(itr) = perfIndex(        W1, ...
                                b1, ...
                                W2, ...
                                b2, ...
                                P,T);
%accumulate normalized gradients       
gW2 = accumulate(S2*A1');
gb2 = accumulate(S2);
gW1 = accumulate(S1*P');
gb1 = accumulate(S1);
%get norm squares
nsgW2o = normSquare(gW2); nsgb2o = normSquare(gb2);
nsgW1o = normSquare(gW1); nsgb1o = normSquare(gb1);
%gradient norms
ngW2o = gW2/sqrt(nsgW2o); ngb2o = gb2/sqrt(nsgb2o);
ngW1o = gW1/sqrt(nsgW1o); ngb1o = gb1/sqrt(nsgb1o);
%initialize directions 
pW1o = gW1; pb1o = gb1; 
pW2o = gW2; pb2o = gb2;
pW1 = -gW1./ngW1o; pb1 = -gb1./ngb1o; 
pW2 = -gW2./ngW2o; pb2 = -gb2./ngb2o;
for i = 2:epochs
    [a, b] = getInterval( @(rate) perfIndex(    W1 + rate*pW1, ...
                                    b1 + rate*pb1, ...
                                    W2 + rate*pW2, ...
                                    b2 + rate*pb2, ...
                                    P,T), startingRate, epsilon);
    rate = golden(@(rate) perfIndex(    W1 + rate*pW1, ...
                                    b1 + rate*pb1, ...
                                    W2 + rate*pW2, ...
                                    b2 + rate*pb2, ...
                                    P,T), a, b);

    itr = itr + 1; 
    sError(itr) = perfIndex(        W1 + rate*pW1, ...
                                    b1 + rate*pb1, ...
                                    W2 + rate*pW2, ...
                                    b2 + rate*pb2, ...
                                    P,T);
    W1 = W1 + rate*pW1;
    b1 = b1 + rate*pb1;
    W2 = W2 + rate*pW2;
    b2 = b2 + rate*pb2;
    %get the first layer output
    A1 = logsigmoid(W1*P + b1);
    %get the second layer output
    A2 = purelin(W2*A1 + b2);
    %error
    E = T - A2;
    %back propigation
    S2 = -2.*E;
    S1 = A1.*(1 - A1).*(W2'*S2);
    %get gradients       
    gW2 = accumulate(S2*A1');
    gb2 = accumulate(S2);
    gW1 = accumulate(S1*P');
    gb1 = accumulate(S1);
    %get norm squares
    nsgW2n = normSquare(gW2);
    nsgb2n = normSquare(gb2);
    nsgW1n = normSquare(gW1);
    nsgb1n = normSquare(gb1);
    %get direction gain
    if mod(itr,resetPeriod) == 0
        bW1 = 0;
        bb1 = 0;
        bW2 = 0;
        bb2 = 0;
    else
        bW1 = nsgW1n/nsgW1o;
        bb1 = nsgb1n/nsgb1o;
        bW2 = nsgW2n/nsgW2o;
        bb2 = nsgb2n/nsgb2o;
    end
    %calculate directions
    pW1n = -gW1 + pW1o*bW1;
    pb1n = -gb1 + pb1o*bb1;
    pW2n = -gW2 + pW2o*bW2;
    pb2n = -gb2 + pb2o*bb2;
    %transfer over old values
    pW1o = pW1n;          pb1o = pb1n;
    pW2o = pW2n;          pb2o = pb2n;
    nsgW2o = nsgW2n;        nsgb2o = nsgb2n;
    nsgW1o = nsgW1n;        nsgb1o = nsgb1n;
    %normalize new direction
    %get norm squares
    nspW2 = normSquare(pW2n);
    nspb2 = normSquare(pb2n);
    nspW1 = normSquare(pW1n);
    nspb1 = normSquare(pb1n);
    %gradient norms
    pW2 = pW2n./sqrt(nspW2);
    pb2 = pb2n./sqrt(nspb2);
    pW1 = pW1n./sqrt(nspW1);
    pb1 = pb1n./sqrt(nspb1);
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

function acc = accumulate(A)
    acc = sum(A,2)./size(A,2);
end

function index = perfIndex(W1, b1, W2, b2, a0, T)
A2 = purelin(W2*logsigmoid(W1*a0 + b1) + b2);
E = (T - A2);
index = E*E';
end

function xn  = normSquare(x)
    xn = sum(sum(x.^2));
end


function out = rangedRand(min, max, R, C)
out = min + (max - min) * rand(R,C);
end