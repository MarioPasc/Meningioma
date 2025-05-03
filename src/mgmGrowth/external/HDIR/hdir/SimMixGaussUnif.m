function [Densities,Responsibilities]=SimMixGaussUnif(Model,Samples)
% Simulate a probabilistic mixture of multivariate Gaussians and uniform or
% triangular distributions.
% The uniform/triangular distribution have a constant support (the support is
% not learnt), which must be a hyperparallelepiped. If we have a triangular
% distribution, it is assumed that all the components are independent and that
% the mode is the center of the hyperparallelepiped.
% Inputs:
%      Model=Model trained by TrainMixGaussUnif.m
%      Samples=Test samples. Each column is a sample
%          (NumSamples x Dimension)
% Outputs:
%      Densities=Vector of size NumSamples x 1 with the probability densities for
%           each test samples: Densities(i)= p( x_i )
%      Responsibilities=Matrix of size NumSamples x NumComp with the
%           each test sample and each mixture component:
%           Responsibilities(i,j)= P( j | x_i )

[Dimension,NumSamples]=size(Samples);

% Find pi_j * p( x_i | j ) for the Gaussian mixture components
Responsibilities=zeros(Model.NumComp,NumSamples);
for NdxCompGauss=1:Model.NumCompGauss
    LogConstant=-0.5*Dimension*log(2*pi)-0.5*log(Model.DetC{NdxCompGauss});
    Differences=Samples-repmat(Model.Mu{NdxCompGauss},1,NumSamples);
    for NdxMuestra=1:NumSamples
        MyResp=Model.Pi(NdxCompGauss)*...
            exp(LogConstant-0.5*Differences(:,NdxMuestra)'*...
            Model.InvC{NdxCompGauss}*Differences(:,NdxMuestra));
        if (isinf(MyResp) || isnan(MyResp))
            Responsibilities(NdxCompGauss,NdxMuestra)=0;
        else
            Responsibilities(NdxCompGauss,NdxMuestra)=MyResp;
        end
    end
end

% Find pi_j * p( x_i | j ) for the uniform/triangular mixture components
if Model.IsTriangular
    % Triangular densities
    for NdxCompUnif=1:Model.NumCompUnif
        % See whether the test samples fall within the support of each
        % mixture component
        WithinLimits=( (Samples<=repmat(Model.Max{NdxCompUnif},1,NumSamples)) & ...
            (Samples>=repmat(Model.Min{NdxCompUnif},1,NumSamples)) );      
        % If it falls within the support, assign the triangular density to
        % it.
        % Otherwise, assign zero density to it.
        Responsibilities(NdxCompUnif+Model.NumCompGauss,:)=...
            (Model.Pi(NdxCompUnif+Model.NumCompGauss)*Model.Den{NdxCompUnif}*...
            prod(1-abs(Samples-repmat(Model.Mode{NdxCompUnif},1,NumSamples))./ ...
            repmat(Model.Half{NdxCompUnif},1,NumSamples),1)).*...
            prod(double(WithinLimits),1);
    end

else
    % Uniform densities
    for NdxCompUnif=1:Model.NumCompUnif
        % See whether the test samples fall within the support of each
        % mixture component
        WithinLimits=( (Samples<=repmat(Model.Max{NdxCompUnif},1,NumSamples)) & ...
            (Samples>=repmat(Model.Min{NdxCompUnif},1,NumSamples)) );      
        % If it falls within the support, assign the uniform density to
        % it.
        % Otherwise, assign zero density to it.
        Responsibilities(NdxCompUnif+Model.NumCompGauss,:)=...
            Model.Pi(NdxCompUnif+Model.NumCompGauss)*Model.Den{NdxCompUnif}*...
            prod(double(WithinLimits),1);
    end    
end

% p(x_i) = sum_over_j { pi_j * p( x_i | j ) }
Densities=sum(Responsibilities,1);

% P( j | x_i ) = { pi_j * p( x_i | j ) } / p( x_i)
Responsibilities=Responsibilities./repmat(Densities,Model.NumComp,1);
        
