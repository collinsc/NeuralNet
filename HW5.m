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
batchSize = 600;
epochs = 3;
idx = randperm(size(P_o,2));
iteration = 0;
disp('    generating weights')
h1 = 300;     %neurons in hidden layer 1
[W1,b1,W2,b2] = getWeights(P_o,T_o, h1);
cW1 = W1; cb1 = b1; cW2 = W2; cb2 = b2;
vW1 = W1; vb1 = b1; vW2 = W2; vb2 = b2;
disp('    generating initial gradient')
idx = randperm(size(P_o,2));
P1 = P_o(:,idx(1:(1+batchSize)-1));
T1 = T_o(:,idx(1:(1+batchSize)-1));
%get initial gradient
[gW1, gb1, gW2, gb2] = accumulateGradients(W1, b1, W2, b2, P1, T1);
pW1 = -gW1;    pb1 = -gb1; 
pW2 = -gW2;    pb2 = -gb2;
disp('Training:')
count = 0;
for i = 1:epochs
    idx = randperm(size(P_o,2));
    P1 = P_o(:,idx(1:(1+batchSize)-1));
    T1 = T_o(:,idx(1:(1+batchSize)-1));
    for batch = batchSize:batchSize:size(P_o,2)   %for each mini batch
        %train on each strategy
        [ cW1, cb1,  cW2,  cb2, ...
           pW1,pb1, pW2, pb2, ...
           gW1, gb1, gW2, gb2, ...
           iteration] = ...
                        conjugateTrain( P1, T1, ...
                                        cW1, cb1, cW2, cb2, ...
                                        pW1, pb1, pW2, pb2, ...
                                        gW1,gb1, gW2, gb2, ...
                                        iteration, 1, true);
        [vW1,vb1,vW2,vb2] = vanillaTrain(   P1, T1, ...
                                                    vW1, vb1, ...
                                                    vW2, vb2, ...
                                                    1, false);
        %test our modifications
        evaluate1 = @(x) evaluate(cW1,cb1,cW2,cb2, x);
        err1 = getPercError(evaluate1,P1,T1);
        evaluate2 = @(x)  evaluate(vW1,vb1,vW2,vb2, x);
        err2 = getPercError(evaluate2,P1,T1);
        count = count +1;
        fprintf('    batch %i\n\tconjugate error percent: %f\n\tvanilla error percent: %f\n', count,err1,err2);
        %grab a batch
        P1 = P_o(:,idx(batch:((batch+batchSize)-1)));
        T1 = T_o(:,idx(batch:((batch+batchSize)-1)));
    end
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
