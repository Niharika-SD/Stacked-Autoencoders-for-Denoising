function DEDiffusion
% this program implement the variant of the PM model in the following paper.
% H. Tian, H. Cai, J. Lai, Effective image noise removal based on difference eigenvalue, ICIP 2011, pp.3357-3360.
% it seems that the result of this model is not satisfactory, however, We think that 
% the strategy of designing an adaptive parameter for the PM model is good, but there is 
% no parameter acting as the "K" parameter in the PM model, I think there should be a similar parameter.
clear all;
close all;
addpath('../images'); 
fpath = '../images/'; 
   

    orgname = 'lung';
%     orgname = 'bacteria'
    orgname = 'lena';
% orgname = 'pepper';
orgname = 'test';
orgname = 'house';
% orgname = 'cameraman';

fname1 = strcat(fpath,orgname); 
fname2 = strcat(fname1,'.bmp');   

dot = max(find(fname2 == '.'));
suffix = fname2(dot+1:dot+3);
if strcmp(suffix,'pgm') | strcmp(suffix,'raw')
    e = rawread(fname2);
else   e = imread(fname2); end
if isrgb(e),  e = rgb2gray(e);  end
if isa(e,'double')~= 1,
    e = double(e);
end
Img = e;

fname = 'lung';
% fname = 'bacteria';
fname = 'lena90';
% fname = 'pepper100';
fname = 'testnoisy100';
fname = 'house20';
% fname = 'cameraman30';

fname = strcat(fname,'.bmp');

img1= imread(fname);
if isrgb(img1),  img1 = rgb2gray(img1);end
if isa(img1,'double')~= 1, img1 = double(img1);end
% input the original image for PSNR calculation...
snro = caclsnr(Img,img1-Img);
[mssim ssim_map] = ssim(Img, img1); 

theta = 1;
dt = 0.1;
noise = 30;

if noise == 100,
    iter = 45500;
    elseif noise == 90,
    iter = 40000; 
    elseif noise == 80,
    iter = 36000; 
    elseif noise == 70,
    iter = 33000; 
    elseif noise == 60,
    iter = 25000;    
elseif noise == 50,
    iter = 20000;
elseif noise == 40.
    iter = 14000;
elseif noise == 30,
    iter = 8000;
elseif noise == 20,
    iter = 5500;
end


K = calcK(img1,theta);

tic;
[J,snr,mssim,i] = DEdiff(img1,iter,K,theta,dt,Img,noise,fname);

 imwrite(uint8(J),strcat( strcat('DEDiff',sprintf('theta[%f] niter[%d]dt[%f],snro[%f],snr[%f] mssim[%f]time[%f]',theta,i,dt,snro,snr,mssim,toc) ),fname ),'bmp');

disp('Over ...\n');

function [J,psnr,mssim,i] = DEdiff(I,iter,K,theta,dt,OImg,noise,fname)
snr0 = 1;% 纪录本次的PSNR
snr00 = 0;% 纪录上次的PSNR
snrflag = 0;% 纪录是否已达到最大的PSNR,if 达到最大,then 写下图像

mssim0 = 1;
mssim00 = 0;% 达到最大即停止

figure;
tic;
[ny,nx]=size(I);

for i=1:iter,  %% do iterations
   % estimate derivatives (Newmann BC)
   K = calcK(I,theta);
   I_mx = I-I(:,[1 1:nx-1]);
   I_px = I(:,[2:nx nx])-I;
   I_my = I-I([1 1:ny-1],:);
   I_py = I([2:ny ny],:)-I;
   KD=K; %(D_ne/max(D_ne(:)));
   
   %pm model function with different k
   Cn=1./(1+(abs(I_my).*KD).^2);
   Cs=1./(1+(abs(I_py).*KD).^2);
   Ce=1./(1+(abs(I_px).*KD).^2);
   Cw=1./(1+(abs(I_mx).*KD).^2);
   
   I_t=-Cn.*I_my + Cs.*I_py + Ce.*I_px - Cw.*I_mx;
     
   I=I+dt*I_t;  %% evolve image by dt   

   pause(0.01);% 
   psnr = caclsnr(OImg,(I)-OImg);
   [mssim ssim_map] = ssim(OImg, (I));  
   fprintf(1,'DEPM: iter -->[%d],psnr[%f],mssim[%f]\n',i,psnr,mssim);
   
   if  mssim0 - mssim00 < 4.0e-5,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.00004,则认为达到最大
         fprintf(1,'mssim0 [%f], mssim00 [%f]\n', mssim0, mssim00);
         if i > 100,
%               break;          
         end
     end
      
      if  snr0 - snr00 < 4.0e-4,% 虽然还在增长，但速度非常缓慢，10次迭代增长小于0.0004,则认为达到最大.
                                % 由于PSNR的值比MSSIM大，因此只精确到这个位数。
          if snrflag == 0,
              fprintf(1,'psnr [%f], snr0 [%f]\n', psnr, snr0);
              snrflag = 1;
          end
      end      
      
     if mod(i,4) ==0,
         if i > 20,
             snr00 = snr0;
             mssim00 = mssim0;
         end
         snr0 = psnr;
         mssim0 = mssim;
     end
     if noise == 100,
    
        if (mod(i,500)==0 && i>15000 && i<28800 )||(mod(i,100)==0 && i>28900 && i<39900) || (mod(i,200)==0 && i>=40000),
               
                  imwrite(uint8(I),strcat( strcat('DEDiff',sprintf('theta[%f] niter[%d]dt[%f],snr[%f] mssim[%f]',theta,i,dt,psnr,mssim) ),fname ),'bmp');
                   close;figure,imshow(uint8(I));
                   title(strcat('DEDiff Image',sprintf('theta[%f], dt [%f] iter [%d] SNR [%f] MSSIM[%f]',theta,dt,i,psnr, mssim) ));  
       
        end
   elseif (noise == 50 || noise == 60 ||noise == 70 ||noise == 80 || noise == 90),  
   
       if (mod(i,500)==0 && i>5000 && i<13800 )||(mod(i,100)==0 && i>13900 && i<15900) || (mod(i,100)==0 && i>=16000),
          
                  imwrite(uint8(I),strcat( strcat('DEDiff',sprintf('theta[%f] niter[%d]dt[%f],snr[%f] mssim[%f]',theta,i,dt,psnr,mssim) ),fname ),'bmp');
                   close;figure,imshow(uint8(I));
                   title(strcat('DEDiff Image',sprintf('theta[%f], dt [%f] iter [%d] SNR [%f] MSSIM[%f]',theta,dt,i,psnr, mssim) ));  
       end
   
elseif noise == 40,
  
      if (mod(i,500)==0 && i>=3000 && i<9000 )||(mod(i,100)==0 && i>9000 && i<11900) || (mod(i,500)==0 && i>=12000),
          
          
                   imwrite(uint8(I),strcat( strcat('DEDiff',sprintf('theta[%f] niter[%d]dt[%f],snr[%f] mssim[%f]',theta,i,dt,psnr,mssim) ),fname ),'bmp');
                   close;figure,imshow(uint8(I));
                   title(strcat('DEDiff Image',sprintf('theta[%f], dt [%f] iter [%d] SNR [%f] MSSIM[%f]',theta,dt,i,psnr, mssim) ));  
          
       end
  
elseif noise == 30,
    
      if (mod(i,300)==0 && i>=2500 && i<=3500 )||(mod(i,100)==0 && i>3500 && i<6900) || (mod(i,100)==0 && i>=7000),
                   imwrite(uint8(I),strcat( strcat('DEDiff',sprintf('theta[%f] niter[%d]dt[%f],snr[%f] mssim[%f]',theta, i,dt,psnr,mssim) ),fname ),'bmp');
                   close;figure,imshow(uint8(I));
                   title(strcat('DEDiff Image',sprintf('theta[%f] dt [%f] iter [%d] SNR [%f] MSSIM[%f]',theta, dt,i,psnr, mssim) ));  
         
       end
  
elseif noise == 20,
    
%        if (mod(k,10)==0 && k>=200 && k<=9900 )||(mod(k,50)==0 && k>9900 && k<19900) || (mod(k,100)==0 && k>=20000),
       if (mod(i,500)==0 && i>=100 && i<=390 )||(mod(i,200)==0 && i>1390 && i<2000) || (mod(i,100)==0 && i>=2000),
%          if (mod(k,10)==0 && k>=500 && k<=990 )||(mod(k,50)==0 && k>1090 && k<5090) || (mod(k,100)==0 && k>=6000),  
                   imwrite(uint8(I),strcat( strcat('DEDiff',sprintf('theta[%f] niter[%d]dt[%f],snr[%f] mssim[%f]',theta,i,dt,psnr,mssim) ),fname ),'bmp');
                   close; figure,imshow(uint8(I));
                   title(strcat('DEDiff Image',sprintf('theta[%f] dt [%f] iter [%d] SNR [%f] MSSIM[%f]',theta,dt,i,psnr, mssim) ));  
        end
     end

end % for i

J = I;
figure,imshow(uint8(J));

return

function K = calcK(im,theta)
sigma = 2.00000001;    % scale parameter in Gaussian kernel
gauss = fspecial('gaussian',round(3*sigma)*2+1,sigma); % Gaussian kernel
f =            BoundMirrorShrink( BoundMirrorShrink( BoundMirrorShrink( BoundMirrorShrink(...
    convbyfft( BoundMirrorExpand( BoundMirrorExpand( BoundMirrorExpand( BoundMirrorExpand( (im) )))),gauss)) )));

[row,col] = size(f);
fx = (f(:,[2:col col]) - f(:,[1 1:col-1]))/2;
fy = (f([2:row row],:) - f([1 1:row-1],:))/2;

fxx = f(:,[2:col col]) + f(:,[1 1:col-1]) - 2.0*f;% second order difference, 2 backward manner
fyy = f([2:row row],:) + f([1 1:row-1],:) - 2.0*f;% second order difference, 2 backward manner
        
Dp  = f([2:row row],[2:col col])  + f([1 1:row-1],[1 1:col-1]);
Dm  = f([1 1:row-1],[2:col col]) + f([2:row row],[1 1:col-1]);
fxy = (Dp-Dm)/4; 
fyx = fxy;

lambda1 = 0.5*(fxx+ fyy + sqrt((fxx - fyy).^2.0+ 4*fxy.^2.0));
lambda2 = 0.5*(fxx+ fyy - sqrt((fxx - fyy).^2.0+ 4*fxy.^2.0));

ff = BoundMirrorExpand(f);

sigma([1:row],[1:col]) = 1/9*( ff([1:row],[1:col]) + ff([1:row],[2:col+1]) + ff([1:row],[3:col+2]) + ...
                               ff([2:row+1],[1:col])                       + ff([2:row+1],[3:col+2]) + ...
                               ff([3:row+2],[1:col]) + ff([3:row+2],[2:col+1]) + ff([3:row+2],[3:col+2])...
                               - 9*ff([2:row+1],[2:col+1]) );
w = theta*normalz(sigma);
p = (lambda1 - lambda2).*lambda1.*w;
K = exp(p/max(p(:)) );

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


function snr1 = caclsnr(I,I0)
snr1 = 10*log10(255*255/mean2((I0).^2));

% PSNR2 = sum(sum( (double(I0)).^2.0) );
% PSNR2 = PSNR2 / prod(size(I));
% 
% PSNR2 = 10*log10(255*255/PSNR2)
