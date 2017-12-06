clear
close all
xrange = [100, 300, 600, 1000];
conjIdx = [2.5, 0.0230, 0.0499, 0.0616];
conjErr = [100, 30, 62.33, 68.1];
vanIdx = [ 1.809e-4, 1.8249e-4, 6.2854e-4, 7.2983e-4];
vanErr = [1,2.33,5.1667,5.2];
figure()
hold on
plot(xrange,conjIdx)
title('Performance Index v. Batch Size')
xlabel('Batch Size')
ylabel('Performance Index')
legend('Conjugate')
hold off
figure()
hold on
plot(xrange,vanIdx)
title('Performance Index v. Batch Size')
xlabel('Batch Size')
ylabel('Performance Index')
legend('Vanilla')
hold off
figure()
hold on
plot(xrange,conjErr)
title('Percent Error v. Batch Size')
xlabel('Batch Size')
ylabel('Percent Error')
legend('Conjugate')
hold off
figure()
hold on
plot(xrange,vanErr)
title('Percent Error v. Batch Size')
xlabel('Batch Size')
ylabel('Percent Error')
legend('Vanilla')
hold off