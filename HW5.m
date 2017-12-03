%reset system state
clear
close all
disp('Load Training Set :')
disp('    loading images')
P_o = loadMNISTImages('./data/train-images-idx3-ubyte');
disp('    loading labels')
%transform to binary output
T_o = formatOutput(loadMNISTLabels('./data/train-labels-idx1-ubyte'));
disp('    getting sets')
%get an order of samples for the batches
batchSize = 256;
epochs = 5;
idx = randperm(size(P_o,2));
P1 = zeros(size(P_o,1), batchSize);
T1 = zeros(size(T_o,1), batchSize);
disp('    generating weights')
h1 = 400;     %neurons in hidden layer 1
[W1,b1,W2,b2] = getWeights(P_o,T_o, h1);
cW1 = W1; cb1 = b1; cW2 = W2; cb2 = b2;
vW1 = W1; vb1 = b1; vW2 = W2; vb2 = b2;
disp('Training:')
count = 0;
for batch = 1:batchSize:size(P_o,2)   %for each mini batch
    %grab a batch
    i = floor(idx(batch)/batchSize)*batchSize +1;
    P1 = P_o(:,i:(i+batchSize)-1);
    T1 = T_o(:,i:(i+batchSize)-1);
    %train on each strategy
    [cW1,cb1,cW2,cb2] = conjugateTrain( P1, T1, ...
                                                cW1, cb1, ...
                                                cW2, cb2, ...
                                                epochs, true);

    [vW1,vb1,vW2,vb2] = vanillaTrain(   P1, T1, ...
                                                vW1, vb1, ...
                                                vW2, vb2, ...
                                                epochs, false);
    %test our modifications
    evaluate1 = @(x) evaluate(cW1,cb1,cW2,cb2, x);
    err1 = getPercError(evaluate1,P1,T1);
    evaluate2 = @(x)  evaluate(vW1,vb1,vW2,vb2, x);
    err2 = getPercError(evaluate2,P1,T1);
    count = count +1;
    fprintf('    batch %i\n\tconjugate error percent: %f\n\tvanilla error percent: %f\n', count,err1,err2);
end
%evaluate network performance
disp('Load Test Set :')
disp('    loading test images')
P2 = loadMNISTImages('./data/t10k-images-idx3-ubyte');
disp('    loading test labels')
T2 = formatOutput(loadMNISTLabels('./data/t10k-labels-idx1-ubyte'));
disp('Run Test Set Conjugate Gradient:')
err = getPercError(evaluate1,P2,T2);
fprintf('    percent error: %f\n',err);
disp('Run Test Set Steepest Descent:')
err = getPercError(evaluate2,P2,T2);
fprintf('    percent error: %f\n',err);
disp('run complete')