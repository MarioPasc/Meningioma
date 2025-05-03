function [Y, Cr, Cb] = RGB2YCC(ImageRGB)
% Color space conversion:
% R'G'B' (nonlinear, gamma corrected) -> JFIF Y_601' Cb Cr, as used in JPEG 
% 256 quantization levels in all color channels

[N M channels] = size(ImageRGB);
Y  = zeros(N, M);
Cr = zeros(N, M);
Cb = zeros(N, M);

ImageRGB=double(ImageRGB);
Y  =  0.299 * ImageRGB(:,:,1) + 0.587 * ImageRGB(:,:,2) + 0.114 * ImageRGB(:,:,3);
Cr = -0.168736 * ImageRGB(:,:,1) - 0.331264 * ImageRGB(:,:,2) + 0.5 * ImageRGB(:,:,3);
Cb =  0.5 * ImageRGB(:,:,1) - 0.418668 * ImageRGB(:,:,2) - 0.081312 * ImageRGB(:,:,3);
