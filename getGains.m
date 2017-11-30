function beta = getGains(gn,g)  
%sgn = accumulate(gn,size(gn,2)); sg = accumulate(g,size(g,2));
% if (~any(sgn)) || (~any(sg))
%     beta = 0;
% else
    beta = normSqr(gn)/normSqr(g);
end