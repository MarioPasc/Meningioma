% HDIR demo
clear all

%FileName=input('Type the input image file name between single quotes: ');
FileName='lena.tif';
%ProportionImpulsiveError=input('Type the proportion of impulsive noise (0 to 1): ');
ProportionImpulsiveError=0.3;
%StdDevGaussianError=input('Type the standard deviation of the Gaussian noise (0 to 100): ');
StdDevGaussianError=25;

randn('state', 0); % initialization
rand('state', 0);

% Load the original image
OrigImg = double(imread(FileName));
[N,M] = size(OrigImg(:,:,1));

% convert the original image into the YCrCb channels
[Y, Cr, Cb] = RGB2YCC(OrigImg);

% Obtain the noisy image in YCrCb
NoisyImgYCrCb = zeros(N, M, 3);
NoisyImgYCrCb(:,:,1) = Y;
% The Cr and Cb channels have minimum -127.5 and maximum 128
NoisyImgYCrCb(:,:,2) = Cr+127.5;
NoisyImgYCrCb(:,:,3) = Cb+127.5;

% Add Gaussian noise
NoisyImgYCrCb = NoisyImgYCrCb + randn(size(NoisyImgYCrCb)) * StdDevGaussianError;
NoisyImgYCrCb(find(NoisyImgYCrCb>255))=255;
NoisyImgYCrCb(find(NoisyImgYCrCb<0))=0;

% Add impulsive noise
BadIndices=find(rand(size(NoisyImgYCrCb))<ProportionImpulsiveError);
NoisyImgYCrCb(BadIndices)=255*rand(size(BadIndices));

% The Cr and Cb channels have minimum -127.5 and maximum 128
NoisyImgYCrCb(:,:,2) = NoisyImgYCrCb(:,:,2)-127.5;
NoisyImgYCrCb(:,:,3) = NoisyImgYCrCb(:,:,3)-127.5;

% The noisy image in RGB
NoisyImg = YCC2RGB(NoisyImgYCrCb(:,:,1), NoisyImgYCrCb(:,:,2), NoisyImgYCrCb(:,:,3));

% Run the HDIR method
[Results]=HDIR(OrigImg,NoisyImg,BadIndices,1);

% Plot results
figure
imshow(OrigImg/255)
hold on
title('Original image')
figure
imshow(NoisyImg/255)
hold on
title('Noisy image')
figure
imshow(Results.ImIRN)
hold on
title('IRN restored image')
figure
imshow(Results.ImHDIR)
title('HDIR restored image')

% Additional information
NoisyImgError=OrigImg-NoisyImg;
disp(sprintf('RMSE of the noisy image: %f\n',sqrt(mean(NoisyImgError(:).^2))))

Results.NoisyImg=NoisyImg;
Results.OrigImg=OrigImg;





