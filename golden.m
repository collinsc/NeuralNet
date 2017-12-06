function out = golden(f,a, b)
tau = 0.618;
tol = 0.0005;
c = a + (1 - tau)*(b - a);
f_c = f(c);
d = b - (1 - tau)*(b - a);
f_d = f(d);
while abs(b - a) > tol
    if (f_c < f_d)
        b = d;
        d = c;
        c = a + (1 - tau)*(b - a);
        f_d = f_c;
        f_c = f(c);
    else
        a = c;
        c = d;
        d = b - (1 - tau)*(b - a);
        f_c = f_d;
        f_d = f(d);
    end
end
out = a;
end