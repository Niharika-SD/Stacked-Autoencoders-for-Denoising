% base code for starting on stacked autoencoders.
% Layers used in this exercise is 2.

%%======================================================================
inputSize = 21*21;
hiddenSizeL1 = 600;    % Layer 1 Hidden Size
hiddenSizeL2 = 500;    % Layer 2 Hidden Size
hiddenSizeL3 = 600;    % Layer 3 Hidden Size 
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 0;              % weight of sparsity penalty term
%%======================================================================
%Load the data for training
%[patches,patches_noise] = sampleIMAGES;

%%======================================================================
%Train the first layer
sae1Theta = initializeParameters(hiddenSizeL1,inputSize);                  %initialize the weights
costFunc = @(p) sparseAutoencoderCost(p,inputSize,hiddenSizeL1,...         %calculate cost and gradient  
                                       lambda,sparsityParam,beta,patches,patches_noise);
options = optimset('MaxIter', 300);
[opttheta1,cost1] = fmincg(costFunc,sae1Theta,options);
%%======================================================================
%Feedforward through the first autoencoder
[sae1Features,W1,b1] = feedForwardAutoencoder(opttheta1, hiddenSizeL1, ...
                                        inputSize, patches);
[sae1Features_noise,~,~] = feedForwardAutoencoder(opttheta1,hiddenSizeL1,inputSize,patches_noise);                                    
%%======================================================================
%Train the second layer
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);               %initialize the weights
costFunc2 = @(p) sparseAutoencoderCost(p,hiddenSizeL1,hiddenSizeL2,...          %calculate cost and gradient  
                                       lambda,sparsityParam,beta,sae1Features,sae1Features_noise);
[opttheta2,cost2] = fmincg(costFunc2,sae2Theta,options);
%%======================================================================
%Feedforward through the second autoencoder
[sae2Features,W2,b2] = feedForwardAutoencoder(opttheta2, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
[sae2Features_noise,~,~] = feedForwardAutoencoder(opttheta2, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features_noise);                                    
%%======================================================================
%Train the third Layer
% sae3Theta = initializeParameters(hiddenSizeL3,hiddenSizeL2);
% costFunc3 = @(p) sparseAutoencoderCost(p,hiddenSizeL2,hiddenSizeL3,...          %calculate cost and gradient  
%                                        lamda,sparsityParam,beta,sae1Features);
% [opttheta3,cost3] = fmincg(costFunc3,sae3Theta,options);                                   
%%======================================================================
%Feedforward through the third autoencoder
% [sae3Features,W3,b3] = feedForwardAutoencoder(opttheta3, hiddenSizeL2, ...
%                                         hiddenSizeL3, patches);
%%======================================================================
%%======================================================================
%Fine Tuning
% Final Layer Training
r  = sqrt(6) / sqrt(hiddenSizeL2+inputSize+1);
W3 = rand(inputSize, hiddenSizeL2) * 2 * r - r;
b3 = zeros(inputSize, 1);
init_theta = [W1(:);W2(:);W3(:);b1(:);b2(:);b3(:)];

costfin = @(p) finetune(p,inputSize,hiddenSizeL1,hiddenSizeL2,...
                                    lambda,patches,patches_noise);
                                
[opttheta,costfinal] = fmincg(costfin,init_theta,options);
%%======================================================================
%Prediction
% testing_stage_exp2.m




%%======================================================================
