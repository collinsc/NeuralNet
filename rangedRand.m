function out = rangedRand(magLow, magHigh, R, C)
out = magLow + (magHigh - magLow) * rand(R,C);
for i =1:size(out,1)
    for j = 1:size(out,2)
       if round(rand) == 1
           out(i,j) = out(i,j)*-1;
       end
    end
end
end