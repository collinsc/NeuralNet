function beta = getGains(gn,g)  
sgn = accumulate(gn,size(gn,2)); sg = accumulate(g,size(g,2));
beta = dot(sgn,sgn)/dot(sg,sg);
end