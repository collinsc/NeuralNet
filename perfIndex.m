function index = perfIndex(W1, b1, W2, b2,W3,b3, P, T)
E = accumulate((T - evaluate(W1,b1,W2,b2,W3,b3,P)),size(T,2));
index = dot(E,E);
end