function [a,b]  = getInterval(f,a,ep)
max_itr = 10;
max_learn = 3;
f_a = f(a);
int = ep;
dec = false;
old = a;
orig = f_a;
for i = 1:max_itr
    b = a + int;
    f_b = f(b);
    if b >= max_learn
      b = a;
      a = old;
      break
    end
    if f_a <= f_b
        if dec
            a = old;
            break
        end
    else
        dec = true;
        old = a;
        a = b;
        f_a = f_b;
    end
    int = int *2;
end
end