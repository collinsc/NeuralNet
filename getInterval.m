function [a,b]  = getInterval(f,a,ep)
max_itr = 10;
f_a = f(a);
for i = 1:max_itr
    b = a+i*ep;
    f_b = f(b);
    %fprintf('a %f f_a %f b %f f_b %f\n', a,f_a, b,f_b);
    if i == max_itr
       disp( 'convergence warn')
    end
    if f_a <= f_b
        if f(b + ep) > f_b
            break
        end
    else
        a = a + i*ep;
        f_a = f_b;
    end
end

end