function [a,b]  = getInterval(f,a,ep)
max_itr = 10;
f_a = f(a);
desc = false;
rate = ep;
for i = 1:max_itr
    b = a+rate;
    f_b = f(b);
    if f_a < f_b
        if desc
            a = old;
            break
        end
        desc = false;
    else
        if ~desc
            desc = true;
            
        end
        old = a;
        a = b;
        f_a = f_b;
    end
    rate = rate * 2;
end

end