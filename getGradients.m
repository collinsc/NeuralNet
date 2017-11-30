function [ gW1, gb1, gW2, gb2, gW3, gb3 ] = getGradients( W1, b1, W2, b2, W3, b3, A0, T )
        %get the first layer output
        A1 = logsigmoid(W1*A0 + b1);
        %get the second layer output
        A2 = logsigmoid(W2*A1 + b2);
        %get the third layer output
        A3 = logsigmoid(W3*A2 + b3);
        %error
        E = T - A3;
        %back propigation
        S3 = -2*diag((1 - A3).*A3)*E;
        S2 = diag((1 - A2).*A2)*(W3'*S3);
        S1 = diag((1 - A1).*A1)*(W2'*S2);
        %accumulate normalized gradients 
        gW1 = S1*(A0');
        gb1 = S1;
        gW2 = S2*(A1');
        gb2 = S2;
        gW3 = S3*(A2');
        gb3 = S3;
end

