figure()
hold on
plot(xrange, cerr, xrange, verr)
legend('conjugate', 'vanilla')
title('Percent Error V. Mini Batch')
xlabel('Mini Batch')
ylabel('Percent Error')
hold off