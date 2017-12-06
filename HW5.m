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
batchSize =100;
epochs = 3;
%stuff for plotting
xrange = 1:((size(P_o,2)*epochs)/batchSize);
cidx = zeros(size(xrange));
cerr = zeros(size(xrange));
verr = zeros(size(xrange));
vidx = zeros(size(xrange));
plotIdx = 1;
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

    for batch = 1:batchSize:size(P_o,2)   %for each mini batch
        P1 = P_o(:,idx(batch:(batch+batchSize)-1));
        T1 = T_o(:,idx(batch:(batch+batchSize)-1));
        %train on each strategy
        [ cW1, cb1,  cW2,  cb2, ...
           pW1,pb1, pW2, pb2, ...
           gW1, gb1, gW2, gb2, ...
           iteration,cid] = ...
                        conjugateTrain( P1, T1, ...
                                        cW1, cb1, cW2, cb2, ...
                                        pW1, pb1, pW2, pb2, ...
                                        gW1,gb1, gW2, gb2, ...
                                        iteration, true);
        cidx(plotIdx) = cid;
        [vW1,vb1,vW2,vb2, vid] = vanillaTrain(   P1, T1, ...
                                                    vW1, vb1, ...
                                                    vW2, vb2, ...
                                                    true);
        vidx(plotIdx) = vid;
        %test our modifications
        evaluate1 = @(x) evaluate(cW1,cb1,cW2,cb2, x);
        cerr(plotIdx) = getPercError(evaluate1,P1,T1);
        evaluate2 = @(x)  evaluate(vW1,vb1,vW2,vb2, x);
        verr(plotIdx) = getPercError(evaluate2,P1,T1);
        count = count +1;
        fprintf('    batch %i\n\tconjugate error percent: %f\n\tvanilla error percent: %f\n', count,cerr(plotIdx),verr(plotIdx));
        plotIdx = plotIdx + 1;
        %let's save our data
    end
end
%evaluate network performance
disp('Load Test Set :')
disp('    loading test images')
P2 = loadMNISTImages('./data/t10k-images-idx3-ubyte');
disp('    loading test labels')
T2 = formatOutput(loadMNISTLabels('./data/t10k-labels-idx1-ubyte'));
evaluate1 = @(x) evaluate(cW1,cb1,cW2,cb2, x);
err1 = getPercError(evaluate1,P1,T1);
evaluate2 = @(x)  evaluate(vW1,vb1,vW2,vb2, x);
err2 = getPercError(evaluate2,P1,T1);
disp('Run Test Set Conjugate Gradient:')
fprintf('    percent error: %f\n',err1);
disp('Run Test Set Steepest Descent:')
fprintf('    percent error: %f\n',err2);
disp('run complete')
plot(xrange, cerr, xrange, cidx, xrange, verr, xrange, vidx)
