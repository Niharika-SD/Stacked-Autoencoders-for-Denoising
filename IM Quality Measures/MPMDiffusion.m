% This code is to implement a Modified PM model (MPM) and the comparison 
%  with You-Kaveh, ROF models for image noise removal.
% References:
% 1. L. Rudin, S. Osher, E. Fatemi,"Nonlinear Total Variation based noise removal algorithms",
%                                   Physica D 60 259-268,1992.
% 2. You,Y., Kaveh, M.: "Fourth Order Partial Differential Equations for Noise Removal", 
%                        IEEE Trans. Image Processing, 2000, vol. 9, no. 10, pp. 1723-1730.
% 3. P. Perona and J. Malik,"Scale-space and edge detection using ansotropic diffusion",
%                    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
%                    12(7):629-639, July 1990.
%
% Yuanquan Wang, 2011/4/19/ 
%==========================================================================
% God bless you, God bless me, God bless us all!
%==========================================================================
function MPMDiffusion(action,varargin)
if nargin<1,
   action='Initialize';
end;
    feval(action,varargin{:});
return;


function Initialize()
global HDmainf;

clear all;
close all;


addpath('../images');

scrsz = get(0,'ScreenSize');
ww = scrsz(3);
hh = scrsz(4);
startp = ww*0.15;
endp = hh*0.15;
fw = scrsz(3)*0.7;
fh = scrsz(4)*0.65; 

HDmainf = figure('Position',[startp endp fw fh],...
    'Color', [0.9 0.9 0.9], ... 
    'NumberTitle', 'off', ...              
    'Name', 'MPM: Modified Perona-Malik model--- A demo', ... 
    'Units', 'pixels');

fpath = '../images/';   


% orgname = 'lung';
orgname = 'bacteria';
% orgname = 'corner20';
orgname = 'lena50';
% orgname = 'testnoisy60';
% orgname = 'pepper100';
orgname = 'house20';
% orgname = 'cameraman30';

fname1 = strcat(fpath,orgname); 
fname = strcat(fname1,'.bmp');   

dot = max(find(fname == '.'));
suffix = fname(dot+1:dot+3);
if strcmp(suffix,'pgm') | strcmp(suffix,'raw')
    e = rawread(fname);
else   e = imread(fname); end
if isrgb(e),  e = rgb2gray(e);  end
if isa(e,'double')~= 1,
    e = double(e);
end
Img =    e;
[ysize xsize] = size(Img);

fpos = get(HDmainf,'Position');
fw = fpos(3); fh = fpos(4);
k = 1;
xpos = 0.5*(fw - xsize*k);
ypos = 0.5*(fh - ysize*k);
HDorigPic1=subplot(1,2,1);  imshow(normalz(e));
title('Original image');  pause(0.001);
% set(HDorigPic1,'Units', 'pixels','Position',[xpos ypos ysize*k xsize*k],'Units','normal'); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
norn = 1;%noise or not noise, if 1, input a noisy image, we need the orginal image to calculate PSNR
%--------------------------------------------------------------------------
if norn == 1,
%     orgn = 'corner';
%     orgname = 'lung';
    orgn = 'bacteria'
    orgn = 'lena';
    orgn = 'house';
%     orgn = 'cameraman';
%     orgn = 'test';
%     orgn = 'pepper';

    fname1 = strcat(fpath,orgn); 
    ffname1 = strcat(fname1,'.bmp');   
    dot = max(find(ffname1 == '.'));
    suffix = ffname1(dot+1:dot+3);
    if strcmp(suffix,'pgm') | strcmp(suffix,'raw')
        e = rawread(ffname1);
    else   e = imread(ffname1); end
    if isrgb(e),  e = rgb2gray(e);  end
    if isa(e,'double')~= 1,
        e = double(e);
    end
    OImg = e;
    I0 = Img;
else 
    % add Gaussian noise
    std_n = 100; % Gaussian noise standard deviation
    Inoise = randn(size(Img))*std_n; % White Gaussian noise
    I0 = Img + Inoise;  % noisy input image
    OImg = Img;
end
%==========================================================================
% Make preparations for the TV, You-Kaveh, and MPM methods, YK model.
%==========================================================================
runflag =  2; %~!~!~ if 0, TV; if 1, You-Kaveh; if 2, MPM, if 3, PM model;
             %     
%--------------------------------------------------------------------------
if runflag == 3,
    I = I0;
else
    I = BoundMirrorExpand(I0);
    I0 = I;
end
subplot(1,2,1);imshow(normalz(Img));title('Original Image');
subplot(1,2,2);imshow(normalz(BoundMirrorShrink(I0)));title('Noisy image');
close;

fname = strcat(orgname,'.bmp');%%%%because the directory .\images\
 
if runflag == 0,
%==========================================================================
% Mehtod 1: run the TV model, a second order PDE model
%==========================================================================
itv = 2000;
dt = 0.1;
lambda = 0.01;
%--------------------------------------------------------------------------
snr0 = 1;% record the PSNR at the current iteration
snr00 = 0;%record the PSNR at the previous iteration
snrflag = 0;% judge if the PSNR is max, if yes, write the image

mssim0 = 1;
mssim00 = 0;% if mssim is max, then stop 
%--------------------------------------------------------------------------
tic;
for k = 1:itv,
%     fprintf(1,'Total Variation: iter -->[%d]\n',k);
%     pause(0.01);
    tv = getTV(I);
    I = I + dt*( tv  - lambda*(I - I0) );
    I = BoundMirrorEnsure(I);
    
    snr = caclsnr(OImg,BoundMirrorShrink(I)-OImg);
    [mssim ssim_map] = ssim(OImg, BoundMirrorShrink(I));  

     fprintf(1,'TV: iter -->[%d],snr[%f],mssim[%f]\n',k,snr,mssim);
     if  mssim0 - mssim00 < 4.0e-4,% if the mssim increase very slowly, say, less than 4.0e-4 after 10 iterations,then we may say mssim is max.
         fprintf(1,'mssim0 [%f], mssim00 [%f]\n', mssim0, mssim00);
         break;          
     end
      
      if  snr0 - snr00 < 4.0e-4,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.001,则认为达到最大.
                                % 由于PSNR的值比MSSIM大，因此只精确到这个位数。          
          if snrflag == 0,
              fprintf(1,'snr [%f], snr0 [%f]\n', snr, snr0);
              snrMax = snr0;
              snrflag = 1;
%               imwrite(normalz(I),strcat( strcat('',sprintf(' PSNR-lapW[%f]kmax[%d]time[%f],snr[%f],mssim[%f],gvc[%d]',lapWeight,k,toc,snr,mssim,gvc)),fname ),'bmp');
          end
      end      
      
     if mod(k,4) ==0,
         if k > 10,
             snr00 = snr0;
             mssim00 = mssim0;
         end
         snr0 = snr;
         mssim0 = mssim;
     end
end
I = BoundMirrorShrink(I);   
snr = caclsnr(OImg,I-OImg);
imwrite(normalz(I),strcat( strcat('TV',sprintf('lambda[%f]niter[%d]dt[%f]time[%f]SNR[%f]mssim[%f]',lambda,k,dt,toc,snr,mssim)),fname ),'bmp');

fprintf(1,'Time elasped for TV: [%f]\n',toc);

figure;imshow(normalz(I));
title(strcat('TV Final Image',sprintf('lambda[%f]niter[%d]dt[%f]time[%f]SNR[%f]mssim[%f]',lambda,k,dt,toc,snr,mssim) ));
disp('for test only!!')
elseif runflag == 1,
%==========================================================================
% Method 2: Fourth order PDE: You and Kaveh model. 
%==========================================================================
iyk = 1000000;
dt = 0.1;
kap = 3;% 3 is good for Noise = 20

snr0 = 1;% 纪录本次的PSNR
snr00 = 0;% 纪录上次的PSNR
snrflag = 0;% 纪录是否已达到最大的PSNR,if 达到最大,then 写下图像

mssim0 = 1;
mssim00 = 0;% 达到最大即停止

tic;
for k = 1:iyk,
%  
    fourthYK = getYK(I,kap);
    I = I + dt*fourthYK;
    I = BoundMirrorEnsure(I);
     snr = caclsnr(OImg,BoundMirrorShrink(I)-OImg);
     [mssim ssim_map] = ssim(OImg, BoundMirrorShrink(I));  
     fprintf(1,'YK: kap-->[%d] iter -->[%d],snr[%f],mssim[%f]\n',kap,k,snr,mssim);
          
     if  mssim0 - mssim00 < 1.0e-5,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.00001,则认为达到最大
         fprintf(1,'mssim0 [%f], mssim00 [%f]\n', mssim0, mssim00);
         if k > 100,
             imwrite(normalz(I),strcat( strcat('',sprintf('YK kap[%d]MSSIM-kmax[%d]time[%f],snr[%f],mssim[%f]',kap,k,toc,snr,mssim)),fname ),'bmp');
             break;          
         end
     end
      
      if  snr0 - snr00 < 1.0e-4,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.0001,则认为达到最大.
                                % 由于PSNR的值比MSSIM大，因此只精确到这个位数。
          if snrflag == 0,
              fprintf(1,'snr [%f], snr0 [%f]\n', snr, snr0);
              snrflag = 1;
              imwrite(normalz(I),strcat( strcat('',sprintf('YK kap[%d]PSNR-kmax[%d]time[%f],snr[%f],mssim[%f]',kap,k,toc,snr,mssim)),fname ),'bmp');
          end
      end      
      
     if mod(k,20) ==0,
         if k > 20,
             snr00 = snr0;
             mssim00 = mssim0;
         end
         snr0 = snr;
         mssim0 = mssim;
     end
end


I = BoundMirrorShrink(I);
fprintf(1,'Total time for You-Kaveh: [%f]\n', toc);
snr = caclsnr(OImg,I-OImg);
% imwrite(normalz(I),strcat( strcat('YK',sprintf('kap[%d]niter[%d]time[%f]snr [%f],mssim[%f]',kap,k,toc,snr,mssim)),fname ),'bmp');

% snr = 10;
figure;imshow(normalz(I)); 
title(strcat('You-Kaveh Final Image',sprintf(' kap [%f], dt [%f] iter [%d] snr [%f]',kap,dt,k,snr) ));
disp('for test only!!')


elseif runflag == 3,
%==========================================================================
% Method 4: PM model
%==========================================================================
niter = 10000;
dt = 0.1;
kappa = 18;
option = 2;

tic;
[diff,i] = anisodiff(I, niter, kappa, dt, option,I0,OImg,fname);
snr = caclsnr(OImg,diff-OImg);

% imwrite(normalz(diff),strcat( strcat('PM',sprintf('kappa[%f]niter[%d]dt[%f]time[%f],snr[%f]',kappa,i,dt,toc,snr)),fname ),'bmp');

fprintf(1,'Time elaspe for PM: [%f]\n',toc);    
%
figure;imshow(normalz(diff));
title(strcat('PM Final Image',sprintf('kappa [%f]  dt [%f] iter [%d] SNR [%f]',kappa,dt,i,snr) ));


elseif runflag == 2,
%==========================================================================
% Method 3: Modified Perona-Malik,referencing the original image
%==========================================================================
impm = 50000;
dt = 0.1;
kappa = 1;

PorE = 0;% if 1, % model 1:Reference, the is noise-free image as reference 采用无噪声参考图像的结构信息
         % else, % model 2:Estimation, if no noise-free image as reference, we have to estimate the image structure from the noisy image 若无无噪声参考图像，则采用噪声图像来估计结构信息 
%==========================================================================
% We may smooth the noisy image using a gaussian of small scale
%==========================================================================
lapWeight = 0.5;% no laplacian regularization
gvc = 1;% if 1, yes, gvc regulariztion, if using reference, 0 is ok;

sigma = 1.000000;    % scale parameter in Gaussian kernel
if PorE ==1,
    [a,c,ac2] = gradAC(BoundMirrorExpand(OImg),sigma, gvc);% model 1:Reference 采用无噪声参考图像的结构信息
else
    [a,c,ac2] = gradAC(I,sigma,gvc);                    % model 2:Estimation 若无无噪声参考图像，则采用噪声图像来估计结构信息
end

%--------------------------------------------------------------------------
snr0 = 1;% 纪录本次的PSNR
snr00 = 0;% 纪录上次的PSNR
snrflag = 0;% 纪录是否已达到最大的PSNR,if 达到最大,then 写下图像

mssim0 = 1;
mssim00 = 0;% 达到最大即停止
%--------------------------------------------------------------------------
tic;
for k = 1:impm,
    pause(0.001);
    
    mpm = getMPM(I,a,c,ac2,kappa);
%      I = I + dt*( mpm );
    I = I + dt*( mpm + lapWeight*getMPMLap(I,kappa) );
    I = BoundMirrorEnsure(I);
    
     snr = caclsnr(OImg,BoundMirrorShrink(I)-OImg);
     [mssim ssim_map] = ssim(OImg, BoundMirrorShrink(I));  
     fprintf(1,'Modified Perona-Malik: iter -->[%d],snr[%f],mssim[%f]\n',k,snr,mssim);
     
     if  mssim0 - mssim00 < 4.0e-4,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.00001,则认为达到最大
         fprintf(1,'mssim0 [%f], mssim00 [%f]\n', mssim0, mssim00);
         break;          
     end
      
      if  snr0 - snr00 < 4.0e-4,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.001,则认为达到最大.
                                % 由于PSNR的值比MSSIM大，因此只精确到这个位数。          
          if snrflag == 0,
              fprintf(1,'snr [%f], snr0 [%f]\n', snr, snr0);
              snrMax = snr0;
              snrflag = 1;
%               imwrite(normalz(I),strcat( strcat('',sprintf(' PSNR-lapW[%f]kmax[%d]time[%f],snr[%f],mssim[%f],gvc[%d]',lapWeight,k,toc,snr,mssim,gvc)),fname ),'bmp');
          end
      end      
      
     if mod(k,4) ==0,
         if k > 10,
             snr00 = snr0;
             mssim00 = mssim0;
         end
         snr0 = snr;
         mssim0 = mssim;
     end
end
I = BoundMirrorShrink(I);   
Iw = I;
snr = caclsnr(OImg,Iw-OImg);
if PorE == 1,
    imwrite(normalz(I),strcat( strcat('',sprintf('Prior MSSIM-lapW[%f]impm[%d]time[%f],snrMax [%f] snr[%f],mssim[%f],gvc[%d]',lapWeight,k,toc,snrMax,snr,mssim, gvc)),fname ),'bmp');
else
    imwrite(normalz(I),strcat( strcat('',sprintf('Est MSSIM-lapW[%f]impm[%d]time[%f],snrMax [%f] snr[%f],mssim[%f],gvc[%d]',lapWeight,k,toc,snrMax,snr,mssim, gvc)),fname ),'bmp');
end

figure;imshow(normalz(Iw));

title(strcat('MPM Final despeckled Image',sprintf(' lapW[%f], dt [%f] iter [%d] SNR [%f] MSSIM[%f],gvc[%d]',lapWeight,dt,k,snr,mssim, gvc) ));

disp('for test only!!')
    
end
%==========================================================================
% EEEEEEEEEEEEEEEENNNNNNNNNNNNNNNNNNNNDDDDDDDDDDDDDDDDDDDDDDDD,END END END
%==========================================================================



function mpmlap = getMPMLap(I,kappa)
I = double(I);
[rows,cols] = size(I);
diff = I;
diffl = zeros(rows+2, cols+2);
diffl(2:rows+1, 2:cols+1) = diff;
Iyc = 0.5*( diffl(3:rows+2,2:cols+1) - diffl(1:rows,2:cols+1) );
Ixc = 0.5*( diffl(2:rows+1,3:cols+2) - diffl(2:rows+1,1:cols) );
Hx =  1./sqrt(1 + (Ixc/kappa).^2 + (Iyc/kappa).^2 );
mpmlap = 4*del2(I).*Hx;   

function mpm = getMPM(I,a,c,ac2,kappa) 
%-------------------------------------------------------------------------
% After trying several convential schemes, e.g., MTV, TV, and ordinary 
% methods for PDEs,we found that all these methods can't yield satisfactory
% results, then we construct one scheme similar to that of the P-M model.
%             
%  (hIx)x(p1)^2.0 + (hIy)y (p2)^2.0 + (hIy)x p1p2 + (hIx)y p1p2
%  
%-------------------------------------------------------------------------

I = double(I);
[rows,cols] = size(I);
diff = I;

diffl = zeros(rows+2, cols+2);
diffl(2:rows+1, 2:cols+1) = diff;

% North, South, East and West differences
  deltaN = diffl(1:rows,2:cols+1)   - diff;% y derivative, forward difference, negative,可以看成当前点的后向差分，从二阶导数的相减来推断
  deltaS = diffl(3:rows+2,2:cols+1) - diff;% y derivative, forward difference, positive,current point
  deltaE = diffl(2:rows+1,3:cols+2) - diff;% x, forward, positive
  deltaW = diffl(2:rows+1,1:cols)   - diff;% x, forward, negative

  Iyc = 0.5*( diffl(3:rows+2,2:cols+1) - diffl(1:rows,2:cols+1) );
  Ixc = 0.5*( diffl(2:rows+1,3:cols+2) - diffl(2:rows+1,1:cols) );

  cN = 1./sqrt(1 + (deltaN/kappa).^2);
  cS = 1./sqrt(1 + (deltaS/kappa).^2);
  cE = 1./sqrt(1 + (deltaE/kappa).^2);
  cW = 1./sqrt(1 + (deltaW/kappa).^2);
  
  mpm1 = (cW.*deltaW + cE.*deltaE).*a.^2.0./ac2 + (cN.*deltaN + cS.*deltaS).*c.^2.0./ac2;
    
  Hx =  1./sqrt(1 + (Ixc/kappa).^2 + (deltaS/kappa).^2 );
  Hy =  1./sqrt(1 + (Iyc/kappa).^2 + (deltaE/kappa).^2 );

HIy = Hy.*Iyc;
HHIy = zeros(rows+2, cols+2);
HHIy(2:rows+1, 2:cols+1) = HIy;
HIyx =  HIy - HHIy(2:rows+1,1:cols);

HIx = Hx.*Ixc;
HHIx = zeros(rows+2, cols+2);
HHIx(2:rows+1, 2:cols+1) = HIx;
HIxy = HIx - HHIx(1:rows,2:cols+1) ;
 
mpm2 = ( HIyx + HIxy ).*a.*c./ac2;
mpm = mpm1 + mpm2;
%--------------------------------------------------------------------------

function [diff,i] = anisodiff(im, niter, kappa, lambda, option,I0,OImg, fname)
mssim0=0;

im = double(im);
[rows,cols] = size(im);
diff = im;
  tic;
for i = 1:niter
  % Construct diffl which is the same as diff but
  % has an extra padding of zeros around it.
  diffl = zeros(rows+2, cols+2);
  diffl(2:rows+1, 2:cols+1) = diff;

  % North, South, East and West differences
  deltaN = diffl(1:rows,2:cols+1)   - diff;% y derivative, forward difference, negative,可以看成当前点的后向差分，从二阶导数的相减来推断
  deltaS = diffl(3:rows+2,2:cols+1) - diff;% y derivative, forward difference, positive,current point
  deltaE = diffl(2:rows+1,3:cols+2) - diff;% x, forward, positive
  deltaW = diffl(2:rows+1,1:cols)   - diff;% x, forward, negative
  
  if option == 1
    cN = exp(-(deltaN/kappa).^2);
    cS = exp(-(deltaS/kappa).^2);
    cE = exp(-(deltaE/kappa).^2);
    cW = exp(-(deltaW/kappa).^2);  
  elseif option == 2
    cN = 1./(1 + (deltaN/kappa).^2);
    cS = 1./(1 + (deltaS/kappa).^2);
    cE = 1./(1 + (deltaE/kappa).^2);
    cW = 1./(1 + (deltaW/kappa).^2);
%    
  end

    diff = diff + lambda*(cN.*deltaN + cS.*deltaS + cE.*deltaE + cW.*deltaW ); 
      snr = caclsnr(OImg,diff-OImg);     
      [mssim ssim_map] = ssim(OImg, diff);  
      
      fprintf(1,' Perona-Malik: iter -->[%d],snr[%f],mssim[%f]\n',i,snr,mssim);     
      if  mssim0 - mssim > 1.0e-7,
          break;          
      end
      mssim0 = mssim;
end
imwrite(normalz(diff),strcat( strcat('PM',sprintf('kappa[%f]iter[%d]lambda[%f]time[%f],snr[%f],mssim[%f]',kappa,i,lambda,toc,snr,mssim)),fname ),'bmp');


function fourthYK = getYK(img,kap)
lapI = 4*del2(img);  % lapI = BoundMirrorEnsure(lapI);
clapI = 1./(1 + (lapI./kap).^2 );
glapI2 = clapI.*lapI;
glapI2 = BoundMirrorEnsure(glapI2);
fourthYK = -4*del2(glapI2);

function kappa = getTV(phi)
%----------------------------------------------------------------
% TV is also the curvature,i.e., div(div(phi)/|div(phi)|),similar
% to the curve length in Chan-Vese active contour model
%----------------------------------------------------------------
[dxf1,dxc1] = fwd_cent_diff(phi');
dxf = dxf1';
dxc = dxc1';
[dyf,dyc] = fwd_cent_diff(phi);
mag1 = sqrt( power(dxf,2.0) + power(dyc,2.0) ) + eps;
dxmag = dxf./mag1;
mag2 = sqrt( power(dyf,2.0) + power(dxc,2.0) ) + eps;
dymag = dyf./mag2;
kappaxb1 = bwd_diff( dxmag');
kappaxb = kappaxb1';
kappayb = bwd_diff( dymag );
kappa = kappaxb + kappayb;
%----------------------------------------------------------------


function [dxf,dxb,dxc] = fwd_bwd_cent_diff(phi)
[m,n] = size(phi);
dxf = zeros(m,n);% dx using forward difference
dxb = zeros(m,n);% dx using backward difference
dxc = zeros(m,n);% dx using central difference
dxf(1:m-1,:) = phi(2:m,:) - phi(1:m-1,:);
dxb(2:m,:) = phi(2:m,:) - phi(1:m-1, :);
dxc(2:m-1,:) = 0.5*(phi(3:m,:) - phi(1:m-2, :));

function [dxf,dxc] = fwd_cent_diff(phi)
[m,n] = size(phi);
dxf = zeros(m,n);% dx using forward difference
dxc = zeros(m,n);% dx using central difference
dxf(1:m-1,:) = phi(2:m,:) - phi(1:m-1,:);
dxc(2:m-1,:) = 0.5*(phi(3:m,:) - phi(1:m-2, :));

function dxf = fwd_diff(phi)
[m,n] = size(phi);
dxf = zeros(m,n);% dx using forward difference
dxf(1:m-1,:) = phi(2:m,:) - phi(1:m-1,:);

function dxb = bwd_diff(phi)
[m,n] = size(phi);
dxb = zeros(m,n);% dx using backward difference
dxb(2:m,:) = phi(2:m,:) - phi(1:m-1, :);

function imgconv = convbyfft(img,g)
[n,m] = size(img);
[k,l] = size(g);
fimg = fft2(img,n+k-1,m+l-1); 
fg = fft2(g,n+k-1,m+l-1);
fimgg = fimg.*fg;
img_temp = real(ifft2(fimgg));
k2 = floor(k/2);
l2 = floor(l/2);
imgconv = img_temp(1+k2:n+k2,1+l2:m+l2);

function [Ixx,Iyy,Iyx,Ixy,Ix,Iy] = order2Diff(I)
[ny,nx] = size(I);
Ix = (I(:,[2:nx nx]) - I(:,[1 1:nx-1]))/2;
Iy = (I([2:ny ny],:) - I([1 1:ny-1],:))/2;

Ixx = I(:,[2:nx nx]) + I(:,[1 1:nx-1]) - 2.0*I;% second order difference, 2 backward manner
Iyy = I([2:ny ny],:) + I([1 1:ny-1],:) - 2.0*I;% second order difference, 2 backward manner
Dp  = I([2:ny ny],[2:nx nx])  + I([1 1:ny-1],[1 1:nx-1]);
Dm  = I([1 1:ny-1],[2:nx nx]) + I([2:ny ny],[1 1:nx-1]);
Ixy = (Dp-Dm)/4; 
Iyx = Ixy;

function [Ixx, Iyy, Iyx,Ixy, Ix, Iy] = minmodDiff(I)
[row,col] = size(I);
Ix = ( I(:,[2:col col]) - I(:,[1 1:col-1]) )/2.0;% central difference 
Iy = ( I([2:row row],:) - I([1 1:row-1],:) )/2.0;% central difference 

Ixp = I(:,[2:col col]) - I(:,:);    % forward difference
Ixx = Ixp(:,:) - Ixp(:,[1 1:col-1]);% backward difference

Iyp = I([2:row row],:) - I(:,:);    % forward difference
Iyy = Iyp(:,:) - Iyp([1 1:row-1],:);% backward difference

Ixn = I(:,:) - I(:,[1 1:col-1]);% backward difference 
Ixminmod = minmod(Ixp,Ixn);     % minmod 
Iyn = I(:,:) - I([1 1:row-1],:);
Iyminmod = minmod(Iyp,Iyn);

Ixy = Ixminmod(:,:) - Ixminmod([1 1:row-1],:); % backward for cross difference
Iyx = Iyminmod(:,:) - Iyminmod(:,[1 1:col-1]); % backward for cross difference

function mmod = minmod(a,b)
mmod = 0.5*(sign(a) + sign(b))*min(abs(a),abs(b));

 
function [a,c,ac2] = gradAC(I,sigma,gvc)
% scale parameter in Gaussian kernel
gauss = fspecial('gaussian',round(3*sigma)*2+1,sigma); % Gaussian kernel
f = BoundMirrorShrink( BoundMirrorShrink( BoundMirrorShrink( BoundMirrorShrink( ...
    convbyfft( BoundMirrorExpand( BoundMirrorExpand( BoundMirrorExpand( BoundMirrorExpand( I)))),gauss)) )));
% figure;imshow(normalz(f));
[dxf1,dxc1] = fwd_cent_diff( f' ); dxc = dxc1';
[dyf,dyc] = fwd_cent_diff( f );
a = -dyc;
c = dxc;

if gvc == 1,
    [ysize xsize] = size(I);  
    rx = floor(xsize/1.0);
    ry = floor(ysize/1.0);
    [Mx,My] = createMask(rx,ry,0);
    a = xconv2(a,Mx);
    c = xconv2(c,My);
end

ac2 = (a.^2.0 + c.^2.0 + eps);
return;

function [Mx,My] = createMask(rx,ry,rz)
Rx = floor(rx*0.5) - 1;
Ry = floor(ry*0.5) - 1;
rz2 = rz*rz;
for i = -Ry:Ry,
    for j = -Rx:Rx,
        if i == 0 & j == 0 ,
            Mx(i+ Ry+1,j+Rx+1) = 0;
            continue;
        end
        Mx(i+ Ry+1,j+Rx+1) = 1/power(sqrt(i*i + j*j + rz2),2.0);
    end
end
My = Mx;

function Y = xconv2(I,G)
[n,m] = size(I);
[n1,m1] = size(G);
FI = fft2(I,n+n1-1,m+m1-1); 
FG = fft2(G,n+n1-1,m+m1-1);
FY = FI.*FG;
YT = real(ifft2(FY));
nl = floor(n1/2);
ml = floor(m1/2);
Y = YT(1+nl:n+nl,1+ml:m+ml);

function snr1 = caclsnr(I,I0)
snr1 = 10*log10(255*255/mean2((I0).^2));

% PSNR2 = sum(sum( (double(I0)).^2.0) );
% PSNR2 = PSNR2 / prod(size(I));
% 
% PSNR2 = 10*log10(255*255/PSNR2)

