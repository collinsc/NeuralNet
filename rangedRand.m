function out = rangedRand(min, max, R, C)
out = min + (max - min) * rand(R,C);
end