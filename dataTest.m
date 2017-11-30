%load test
% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images1 = loadMNISTImages('./data/t10k-images-idx3-ubyte');
labels1 = loadMNISTLabels('./data/t10k-labels-idx1-ubyte');
images2 = loadMNISTImages('./data/train-images-idx3-ubyte');
labels2 = loadMNISTLabels('./data/train-labels-idx1-ubyte');
% We are using display_network from the autoencoder code
display_network(images1(:,1:100)); % Show the first 100 images
disp(labels1(1:10));
%We are using display_network from the autoencoder code
display_network(images2(:,1:100)); % Show the first 100 images
disp(labels2(1:10));