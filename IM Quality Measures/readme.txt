DEDiffusion: this program implement the variant of the PM model in the following paper.
H. Tian, H. Cai, J. Lai, Effective image noise removal based on difference eigenvalue, ICIP 2011, pp.3357-3360.
It seems that the result of this model is not satisfactory; however, We think that the strategy of designing an adaptive parameter for the PM model is graceful, but there is no parameter acting as the "K" parameter in the PM model, I think there should be a similar parameter, and if there is a similar “K” parameter in the proposed model, the results would be very good.

MPMDiffusion: in this program, we implemented an modified Perona-Malik  model using the directional Laplacian. 
Yuanquan Wang, J.C. Guo, W.F. Chen and Wenxue Zhang, Image denoising using modified Perona-Malik model based on directional Laplacian, Signal Processing, Volume 93, Issue 9, September 2013, Pages 2548-2558 

The contribution of this paper is 3folded: (1) we reformulate the PM model using the directional Laplacian and proposed a novel model for image noise removal. (2) it is interesting that the famous TV model can also be formulated using the directional Laplacian.(3) the proposed model can preserve smoothly-varying surface and edges. 
However, to be frank, the proposed model cannot yield results as good as the patch-based methods, such as the nonlocal mean,BM3D, PLOW/LARK etc by Milanfar etc, and also the sparse representation based methods. However, it advances the development of the PDE-based methods for image restoration, and I think our major contribution is theoretical. As far as I know, the local PDE based method always performs inferior to the nonlocal method.

Also, the PM, TV and YK models are also implemented, and the MSSIM program is borrowed.

