%evaluate the 3 layer logsig neural network
function [ out ] = evaluate( W1, b1, W2, b2, W3, b3, x)
out = logsigmoid(W3*logsigmoid(W2*logsigmoid(W1*x+b1) + b2) + b3);
end

