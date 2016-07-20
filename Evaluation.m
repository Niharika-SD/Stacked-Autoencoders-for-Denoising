%%Use simililarity measures to evaluate performance
%% PSNR calculation

addpath('/nfs4/gautam/Documents/MATLAB/Stacked Autoencoders for Denoising/')

test_psnr.avgpatch = PSNR(im2double(mat2gray(imread('avg_patch_caseRes.tif'))),im2double(mat2gray(imread('original.tif'))));
test_psnr.avgpatch_r = PSNR(im2double(mat2gray(imread('avg_patch_caseRes_rep.tif'))),im2double(mat2gray(imread('original.tif'))));

test_psnr.bavg = PSNR(im2double(mat2gray(imread('blockavg_caseRes.tif'))),im2double(mat2gray(imread('original.tif'))));
test_psnr.bavg_r = PSNR(im2double(mat2gray(imread('blockavg_caseRes_rep.tif'))),im2double(mat2gray(imread('original.tif'))));

test_psnr.addorig = PSNR(im2double(mat2gray(imread('Addorig_caseRes.tif'))),im2double(mat2gray(imread('original.tif'))));
test_psnr.addorig_r = PSNR(im2double(mat2gray(imread('Addorig_caseRes_rep.tif'))),im2double(mat2gray(imread('original.tif'))));

test_pnsr.corr = PSNR(im2double(mat2gray(imread('corrupted.tif'))),im2double(mat2gray(imread('original.tif'))));

%% SSIM calculation
test_ssim.avgpatch = ssim(im2double(mat2gray(imread('avg_patch_caseRes.tif'))),im2double(mat2gray(imread('original.tif'))));
test_ssim.avgpatch_r = ssim(im2double(mat2gray(imread('avg_patch_caseRes_rep.tif'))),im2double(mat2gray(imread('original.tif'))));

test_ssim.bavg = ssim(im2double(mat2gray(imread('blockavg_caseRes.tif'))),im2double(mat2gray(imread('original.tif'))));
test_ssim.bavg_r = ssim(im2double(mat2gray(imread('blockavg_caseRes_rep.tif'))),im2double(mat2gray(imread('original.tif'))));

test_ssim.addorig = ssim(im2double(mat2gray(imread('Addorig_caseRes.tif'))),im2double(mat2gray(imread('original.tif'))));
test_ssim.addorig_r = ssim(im2double(mat2gray(imread('Addorig_caseRes_rep.tif'))),im2double(mat2gray(imread('original.tif'))));

test_ssim.corr = ssim(im2double(mat2gray(imread('corrupted.tif'))),im2double(mat2gray(imread('original.tif'))));