function err = getPercError(f,P,T)
A = round(f(P));
errCount = 0;
for i = 1:size(P,2)
    if sum((T(:,i)-A(:,i)).^2)> 0
        errCount = errCount + 1;
    end
end
err = errCount./size(T,2)*100;
end