%evaluate the 2 layer logsig neural network
function [ out ] = evaluate( W1, b1, W2, b2, x)
out = zeros(size(b2,1),size(x,2));
for i = 1:size(x,2)
    out(:,i) = logsigmoid(W2*logsigmoid(W1*x(:,i)+b1) + b2);
end

