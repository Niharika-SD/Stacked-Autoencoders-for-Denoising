load patches.mat
for i=1:size(patches,2)
    patch = reshape(patches(:,i),[21,21]);
    imshow(patch);
    patch = imnoise(patch,'gaussian');
    patches_noise(:,i) = reshape(patch,[441,1]);
    i
end
save('patches_noise.mat','patches_noise')