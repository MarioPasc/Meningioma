function ExecHDIR_MRI(inputImagePath, outputMatrixPath)
% ExecHDIR_MRI: Function to compute and save the responsibilities matrix for MRI images

% Clear workspace and command window (optional)
clc;

% Load the input image
fprintf('Loading image from: %s\n', inputImagePath);
NoisyImg = imread(inputImagePath);
fprintf('Image size: %s\n', mat2str(size(NoisyImg)));

% Ensure the image is in double format
NoisyImg = double(NoisyImg);

% Validate the input image is single-channel
if size(NoisyImg, 3) ~= 1
    error('Input image must be single-channel (grayscale).');
end

% Call HDIR_MRI to compute and save the responsibilities matrix
fprintf('Processing the image to compute responsibilities...\n');
SaveResponsibilities = HDIR_MRI(NoisyImg, outputMatrixPath);

% Confirm completion
fprintf('Responsibilities computation complete.\n');
fprintf('Responsibilities matrix saved to: %s\n', outputMatrixPath);
end
