%evaluate the 2 layer logsig neural network
function [ out ] = evaluate( W1, b1, W2, b2, x)
out = logsigmoid(W2*logsigmoid(W1*x+b1) + b2);
end

