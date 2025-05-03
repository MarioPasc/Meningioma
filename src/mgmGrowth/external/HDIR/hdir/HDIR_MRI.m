function SaveResponsibilities = HDIR_MRI(NoisyImg, SavePath)
% HDIR_MRI: Computes and saves the responsibilities matrix for MRI images.
% The responsibilities represent the posterior probabilities of impulse noise corruption.
%
% Inputs:
%   NoisyImg: Single-channel noisy MRI image (grayscale).
%   SavePath: Path to save the responsibilities matrix (e.g., 'responsibilities.mat').
%
% Outputs:
%   SaveResponsibilities: Matrix of posterior probabilities of impulse noise corruption.
%   Saves the matrix to the specified path.

    % Parameters for noise model training
    LimitsUnif.Min = { -256 }; % Minimum value for the uniform noise
    LimitsUnif.Max = { 256 }; % Maximum value for the uniform noise

    % Step 1: Error Estimation using IRN
    lambda  = 1.2;
    pars = irntvInputPars('l1tv');
    pars.pcgtol_ini   = 1e-4;
    pars.adapt_epsR   = 1;
    pars.epsR_cutoff  = 0.01;
    pars.adapt_epsF   = 1;
    pars.epsF_cutoff  = 0.05;
    pars.loops        = 3;

    % Apply IRN to estimate the reconstructed image
    [ImAux, ~] = irntv(NoisyImg, [], lambda, pars);

    % Compute the error between reconstructed and noisy image
    EstimError = ImAux - NoisyImg;

    % Step 2: Train the Noise Model
    % Subsample errors for training
    Samples = EstimError(1:20:end);

    % Train the Gaussian + Uniform mixture model
    Model = TrainMixGaussUnif(Samples, 1, 1, LimitsUnif, 1);

    % Step 3: Simulate the Model and Compute Responsibilities
    % Compute responsibilities (posterior probabilities)
    [~, Responsibilities] = SimMixGaussUnif(Model, EstimError(:)');
    
    fprintf('Responsibilities pre-reshape size: %s\n', mat2str(size(Responsibilities)));
    
    % Extract the impulse noise probabilities (row 2) and reshape
    SaveResponsibilities = reshape(Responsibilities(2, :), size(NoisyImg));
    
    fprintf('Responsibilities post-reshape size: %s\n', mat2str(size(SaveResponsibilities)));


    % Normalize to ensure values are between [0, 1]
    SaveResponsibilities = min(max(SaveResponsibilities, 0), 1);

    % Step 4: Save the Responsibilities Matrix
    save(SavePath, 'SaveResponsibilities');
    fprintf('Responsibilities matrix saved to: %s\n', SavePath);
end
