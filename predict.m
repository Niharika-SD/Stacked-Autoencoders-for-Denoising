function [activation] = predict(theta, hiddenSizeL1,hiddenSizeL2, visibleSize, data)

q = (hiddenSizeL1*hiddenSizeL2)+(hiddenSizeL1*visibleSize);
q1 = (hiddenSizeL2*visibleSize); 
W1 = reshape(theta(1:hiddenSizeL1*visibleSize), hiddenSizeL1, visibleSize);
W2 = reshape(theta(hiddenSizeL1*visibleSize+1:q), hiddenSizeL2, hiddenSizeL1);
W3 = reshape(theta(q+1:q+q1),visibleSize,hiddenSizeL2);
b1 = theta(q+q1+1:q+q1+hiddenSizeL1);
b2 = theta(q+q1+hiddenSizeL1+1:q+q1+hiddenSizeL1+hiddenSizeL2);
b3 = theta(q+q1+hiddenSizeL1+hiddenSizeL2+1:end);

[~,~,~,activation] = getActivation(W1,W2,W3,b1,b2,b3,data);
% patch_out = reshape(activation,[21,21]);


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
function [ainput, ahidden1,ahidden2, aoutput] = getActivation(W1, W2,W3, b1, b2,b3, data)
 
ainput = data;
ahidden1 = bsxfun(@plus, W1 * ainput, b1);
ahidden1 = sigmoid(ahidden1);
ahidden2 = bsxfun(@plus, W2 * ahidden1, b2);
ahidden2 = sigmoid(ahidden2);
aoutput = bsxfun(@plus, W3 * ahidden2, b3);
aoutput = sigmoid(aoutput);
end
end