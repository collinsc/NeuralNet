function out = golden(f,a, b)
tau = 0.618;
tol = 0.0005;
c = a + (1 - tau)*(b - a);
f_c = f(c);
d = b - (1 - tau)*(b - a);
f_d = f(d);
while b - a > tol
    %debugging
    %fprintf('a %f b %f c %f d %f f_c %f, f_d %f\n', a, b,c, d, f_c,f_d)
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