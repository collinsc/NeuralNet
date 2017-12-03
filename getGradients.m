function [ gW1, gb1, gW2, gb2] = getGradients( W1, b1, W2, b2, A0, T )
        %get the first layer output
        A1 = logsigmoid(W1*A0 + b1);
        %get the second layer output
        A2 = logsigmoid(W2*A1 + b2);
        %error
        E = T - A2;
        %back propigation
        S2 = -2*diag((1 - A2).*A2)*E;
        S1 = diag((1 - A1).*A1)*(W2'*S2);
        %accumulate normalized gradients 
        gW1 = S1*(A0');
        gb1 = S1;
        gW2 = S2*(A1');
        gb2 = S2;
end

