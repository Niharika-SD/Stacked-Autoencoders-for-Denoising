srcFiles = dir('C:\Users\Gautam Sridhar\Documents\MATLAB\Rainstreak\Train_data\Train_real\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('C:\Users\Gautam Sridhar\Documents\MATLAB\Rainstreak\Train_data\Train_real\',srcFiles(i).name);
    I = imread(filename);
    I = rgb2gray(I);
    I = im2double(I);
    imshow(I);
    sample_images(:,:,i) = I;
%   sample_images_noise(:,:,i) = imnoise(I,'gaussian');
    
end
save ('sample_images.mat','sample_images');
% srcFiles1 = dir('C:\Users\Gautam Sridhar\Documents\MATLAB\Rainstreak\Train_data\Train_rainstreak\*.jpg');  % the folder in which ur images exists
% for i = 1 : length(srcFiles1)
%     filename = strcat('C:\Users\Gautam Sridhar\Documents\MATLAB\Rainstreak\Train_data\Train_rainstreak\',srcFiles1(i).name);
%     I = imread(filename);
%     I = rgb2gray(I);
%     I = im2double(I);
%     imshow(I);
%     sample_images_noise(:,:,i) = I; 
% end
% save ('sample_images_noise.mat','sample_images_noise');
