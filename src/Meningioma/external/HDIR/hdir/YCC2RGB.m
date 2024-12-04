function ImageRGB = YCC2RGB(Y, Cr, Cb)
% Color space conversion:
% JFIF Y_601' Cb Cr, as used in JPEG -> R' G' B' (nonlinear, gamma
% corrected)
% 256 quantization levels in all color channels

[N M] = size(Y);
ImageRGB = zeros(N, M, 3);

ImageRGB(:,:,1) = Y - 0.00092460 * Cr + 1.40168676 * Cb;
ImageRGB(:,:,2) = Y - 0.34369538 * Cr - 0.71416904 * Cb;
ImageRGB(:,:,3) = Y + 1.77216042 * Cr + 0.00099022 * Cb;
