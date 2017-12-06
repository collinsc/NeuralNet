figure()
hold on
plot(xrange, cidx, xrange, vidx)
legend('conjugate', 'vanilla')
title('Performance Index V. Mini Batch')
xlabel('Mini Batch')
ylabel('Performance Index')
hold off