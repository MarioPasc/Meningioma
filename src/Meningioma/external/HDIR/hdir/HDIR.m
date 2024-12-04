function [Results]=HDIR(OrigImg,NoisyImg,BadIndices,PlotIt)
% Highly Damaged Image Reconstructor
% Inputs:
%   OrigImg=Original image
%   NoisyImg=Noisy image
%   BadIndices=Indices of the impulse noise corrupted pixels
%   PlotIt=Flag to generate plots
% Outputs:
%   Results.ImIRN=Reconstructed image by Iteratively Reweighted Norm
%   (Total Variation), L1 version
%   Results.ImHDIR=Reconstructed image by Highly Damaged Image
%       Reconstructor
%   Results.RMSE_IRN, Results.RMSE_HDIR=Root Mean Squared Error
%       for IRN and HDIR
%   Results.StdNoise=Standard deviation of the Gaussian error, estimated by
%       HDIR
%   Results.PrImpul=Proportion of impulse noise, estimated by HDIR


[N,M] = size(OrigImg(:,:,1));

% Convert the noisy image from RGB to YCrCb
[Y, Cr, Cb] = RGB2YCC(NoisyImg);

BadImg = zeros(N, M, 3);
BadImg(:,:,1) = Y;
BadImg(:,:,2) = Cr;
BadImg(:,:,3) = Cb;

% Apply IRN: Iteratively Reweighted Norm (Total Variation), L1 version

lambda  = 1.2;
pars = irntvInputPars('l1tv');
pars.pcgtol_ini   = 1e-4;
pars.adapt_epsR   = 1;
pars.epsR_cutoff  = 0.01;
pars.adapt_epsF   = 1;
pars.epsF_cutoff  = 0.05;
pars.loops        = 3;

[ImAuxY,itstatus] = irntv(Y, [], lambda, pars);
[ImAuxCr,itstatus] = irntv(Cr, [], lambda, pars);
[ImAuxCb,itstatus] = irntv(Cb, [], lambda, pars);

Results.ImIRN=YCC2RGB(ImAuxY,ImAuxCr,ImAuxCb);
MyError=OrigImg-Results.ImIRN;
Results.RMSE_IRN=sqrt(mean(MyError(:).^2));

% HDIR error estimation

EstimError1=squeeze(ImAuxY-BadImg(:,:,1));
EstimError2=squeeze(ImAuxCr-BadImg(:,:,2));
EstimError3=squeeze(ImAuxCb-BadImg(:,:,3));

Samples1=EstimError1(1:20:end);
Samples2=EstimError2(1:20:end);
Samples3=EstimError3(1:20:end);

LimitsUnif.Min{1}=[ -256 ]; 
LimitsUnif.Max{1}=[ 256 ];

% Triangular distribution training
Model1=TrainMixGaussUnif(Samples1,1,1,LimitsUnif,1);
[Densities1,Responsibilities1]=SimMixGaussUnif(Model1,EstimError1(:)');
I1=reshape(Responsibilities1(1,:),size(BadImg,1),size(BadImg,2));

Model2=TrainMixGaussUnif(Samples2,1,1,LimitsUnif,1);
[Densities2,Responsibilities2]=SimMixGaussUnif(Model2,EstimError2(:)');
I2=reshape(Responsibilities2(1,:),size(BadImg,1),size(BadImg,2));

Model3=TrainMixGaussUnif(Samples3,1,1,LimitsUnif,1);
[Densities3,Responsibilities3]=SimMixGaussUnif(Model3,EstimError3(:)');
I3=reshape(Responsibilities3(1,:),size(BadImg,1),size(BadImg,2));

% Apply classic kernel regression
h = 2.3;    % Global smoothing parameter
ksize = 15; % Kernel size
[zc, zx1c, zx2c] = ClassicKernelRegression(BadImg(:,:,1), I1, h, ksize);

% Compute steering matrices
wsize = 9;   % Size of the local analysis window
lambda = 1;  % Tegularization for the elongation parameter
alpha = 0.1; % Structure sensitive parameter
C = SteeringMatrix(zx1c, zx2c, I1, wsize, lambda, alpha);

% Apply steering kernel regression
[zs, zx1s, zx2s] = SteeringKernelRegression(BadImg(:,:,1), I1, h, C, ksize);
z(:,:,1)=zs;
[zs, zx1s, zx2s] = SteeringKernelRegression(BadImg(:,:,2), I2, h, C, ksize);
z(:,:,2)=zs;
[zs, zx1s, zx2s] = SteeringKernelRegression(BadImg(:,:,3), I3, h, C, ksize);
z(:,:,3)=zs;

% Store results and compute RMSE
Results.ImHDIR = YCC2RGB(z(:,:,1), z(:,:,2), z(:,:,3));
MyError=OrigImg-Results.ImHDIR;
Results.RMSE_HDIR=sqrt(mean(MyError(:).^2));

Results.StdNoise=sqrt(mean([Model1.C{1} Model2.C{1} Model3.C{1}]));
Results.PrImpul=mean([Model1.Pi(2) Model2.Pi(2) Model3.Pi(2)]);

% Plot additional information if desired
if PlotIt
    [DensitiesDib,ResponsibilitiesDib]=SimMixGaussUnif(Model1,-255:255);
    figure;
    Values=hist(EstimError1(:),-260:4:260);
    H1=bar(-260:4:260,Values,1,'c');
    hold on;
    H2=plot(-255:255,4*M*N*DensitiesDib,'-r');
    xlabel('Estimated pixel error')
    ylabel('Number of pixels')
    legend([H1 H2],'Real','Model');
    hold off;
    [Y, Cr, Cb] = RGB2YCC(OrigImg);
    MyError=Y-BadImg(:,:,1);
    figure;
    Values=hist(MyError(:),-260:4:260);
    H1=bar(-260:4:260,Values,1,'c');
    hold on;
    H2=plot(-255:255,4*M*N*DensitiesDib,'-r');
    xlabel('Pixel error');
    ylabel('Number of pixels');
    legend([H1 H2],'Real','Model');
    hold off;

    BadIndicesY=BadIndices(find(BadIndices<=M*N));
    ImpulseErrors=MyError(BadIndicesY);
    IndicesBuenos=setdiff(1:(M*N),BadIndices);
    NonImpulseErrors=MyError(IndicesBuenos);
    HistBads=hist(ImpulseErrors,-260:4:260);
    HistGoods=hist(NonImpulseErrors,-260:4:260);
    PropImpulses=HistBads./(HistBads+HistGoods);
    figure;
    H1=plot(-260:4:260,PropImpulses,'-g');
    hold on;
    H2=plot(-255:255,ResponsibilitiesDib(2,:),'-r');
    xlabel('Pixel error');
    ylabel('Probability of being an impulse-corrupted pixel');
    legend([H1 H2],'Real','Model');
    hold off;
end

Results.ImHDIR=uint8(Results.ImHDIR);
Results.ImIRN=uint8(Results.ImIRN);

