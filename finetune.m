function [cost,grad] = finetune(theta,visibleSize,hiddenSizeL1,hiddenSizeL2,...
                                    lambda,data_clean,data_noise)

q = (hiddenSizeL1*hiddenSizeL2)+(hiddenSizeL1*visibleSize);
q1 = (hiddenSizeL2*visibleSize); 
W1 = reshape(theta(1:hiddenSizeL1*visibleSize), hiddenSizeL1, visibleSize);
W2 = reshape(theta(hiddenSizeL1*visibleSize+1:q), hiddenSizeL2, hiddenSizeL1);
W3 = reshape(theta(q+1:q+q1),visibleSize,hiddenSizeL2);
b1 = theta(q+q1+1:q+q1+hiddenSizeL1);
b2 = theta(q+q1+hiddenSizeL1+1:q+q1+hiddenSizeL1+hiddenSizeL2);
b3 = theta(q+q1+hiddenSizeL1+hiddenSizeL2+1:end);

W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
W3grad = zeros(size(W3));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
b3grad = zeros(size(b3));

[~, nSamples] = size(data_noise);

[a1, a2, a3,a4] = getActivation(W1, W2,W3, b1, b2,b3, data_noise);
errtp = ((a4 - data_clean) .^ 2) ./ 2;
err = sum(sum(errtp)) ./ nSamples;

err2 = sum(sum(W3 .^ 2));
err2 = err2 * lambda / 2;

cost = err + err2;

delta4 = -(data_clean - a4) .* dsigmoid(a4);
delta3 = (W3' * delta4); 
delta3 = delta3 .* dsigmoid(a3);
delta2 = (W2' * delta3);
delta2 = delta2 .* dsigmoid(a2);
nablaW1 = delta2 * a1';
nablab1 = delta2;
nablaW2 = delta3 * a2';
nablab2 = delta3;
nablaW3 = delta4 * a3';
nablab3 = delta4;
 
W1grad = nablaW1 ./ nSamples;
W2grad = nablaW2 ./ nSamples;
W3grad = nablaW3 ./ nSamples + lambda.*W3;
b1grad = sum(nablab1, 2) ./ nSamples;
b2grad = sum(nablab2, 2) ./ nSamples;
b3grad = sum(nablab3, 2) ./ nSamples;

grad = [W1grad(:);W2grad(:);W3grad(:);b1grad(:);b2grad(:);b3grad(:)];
end


function sigm = sigmoid(x)
 
sigm = 1 ./ (1 + exp(-x));
end

function dsigm = dsigmoid(a)
dsigm = a .* (1.0 - a);
 
end
function [ainput, ahidden1,ahidden2, aoutput] = getActivation(W1, W2,W3, b1, b2,b3, input)
 
ainput = input;
ahidden1 = bsxfun(@plus, W1 * ainput, b1);
ahidden1 = sigmoid(ahidden1);
ahidden2 = bsxfun(@plus, W2 * ahidden1, b2);
ahidden2 = sigmoid(ahidden2);
aoutput = bsxfun(@plus, W3 * ahidden2, b3);
aoutput = sigmoid(aoutput);
end