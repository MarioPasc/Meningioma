function [Model]=TrainMixGaussUnif(Samples,NumCompGauss,NumCompUnif,LimitsUnif,IsTriangular)
% Train a probabilistic mixture of multivariate Gaussians and uniform or
% triangular distributions.
% The uniform/triangular distribution have a constant support (the support is
% not learnt), which must be a hyperparallelepiped. If we have a triangular
% distribution, it is assumed that all the components are independent and that
% the mode is the center of the hyperparallelepiped.
% Inputs:
%   Samples=Training samples. Each column is a sample
%          (NumSamples x Dimension)
%   NumCompGauss=Number of Gaussian mixture components
%   NumCompUnif=Number of uniform/triangular mixture components
%   LimitsUnif=Structure which contains the boundaries of the support of
%       each uniform mixture component. We have that LimitsUnif.Min{ndx} is
%       a column vector with size Dimension x 1 which contains the minimums
%       for each input space dimension, corresponding to the ndx-th
%       uniform/triangular mixture component. LimitsUnif.Max{ndx} is
%       a column vector with size Dimension x 1 which contains the maximums
%       for each input space dimension, corresponding to the ndx-th
%       uniform/triangular mixture component.
%   IsTriangular=0 for uniform distributions, =1 for triangular distributions
% Output:
%   Model=Mixture model trained by Expectation Maximization (EM). The
%     fields are:
%       Model.Mu{ndx} -> Mean vector of the ndx-th Gaussian mixture
%           component
%       Model.C{ndx} -> Covariance matrix of the ndx-th Gaussian mixture
%           component
%       Model.Den{ndx} -> Probability density of the ndx-th uniform mixture
%           component within its support
%       Model.Min{ndx},Model.Max{ndx} -> Same as LimitsUnif
%       Model.Pi(ndx) -> A priori probability of the ndx-th mixture
%           component. The Gaussian components appear first. Model.Pi is of
%           size NumCompGauss+NumCompUnif


[Dimension,NumSamples]=size(Samples);

Model=LimitsUnif;
Model.Pi(1:NumCompGauss,1)=ones(NumCompGauss,1)*(0.9/NumCompGauss);
Model.Pi(end+1:end+NumCompUnif,1)=ones(NumCompUnif,1)*(0.1/NumCompUnif);
Model.NumCompGauss=NumCompGauss;
Model.NumCompUnif=NumCompUnif;
Model.NumComp=NumCompGauss+NumCompUnif;
Model.IsTriangular=IsTriangular;

% Find uniform/triangular probability densities
if IsTriangular
    % Store the maximum density,
    % the mode of each mixture component
    % and half the length of the support in each component
    for NdxComp=1:NumCompUnif
        Model.Den{NdxComp}=prod(2./(Model.Max{NdxComp}-Model.Min{NdxComp}));
        Model.Mode{NdxComp}=0.5*(Model.Max{NdxComp}+Model.Min{NdxComp});
        Model.Half{NdxComp}=0.5*(Model.Max{NdxComp}-Model.Min{NdxComp});
    end
else    
    for NdxComp=1:NumCompUnif
        Volume=prod(Model.Max{NdxComp}-Model.Min{NdxComp});
        Model.Den{NdxComp}=1/Volume;
    end
end


% Initialize Gaussian mixture components
for NdxComp=1:NumCompGauss
    MySamples=Samples.*(1+0.01*rand(size(Samples)));    
    Model.Mu{NdxComp}=Samples(:,ceil(NumSamples*rand(1)));
    Model.C{NdxComp}=cov(MySamples');
    Model.InvC{NdxComp}=inv(Model.C{NdxComp});
    Model.DetC{NdxComp}=det(Model.C{NdxComp});
end

% EM algorithm (iterative)
ContinueIt=1; 
while ContinueIt
    AntPi=Model.Pi;
    % Find responsibilities
    [Densities,Responsibilities]=SimMixGaussUnif(Model,Samples);
       
    % Update a priori probabilities
    Model.Pi=sum(Responsibilities,2)/NumSamples;
    
    % Update the parameters of the Gaussian mixture components
    for NdxComp=1:NumCompGauss
        Model.Mu{NdxComp}=sum(repmat(Responsibilities(NdxComp,:),Dimension,1).* ...
            Samples,2)/(NumSamples*Model.Pi(NdxComp));
        Differences=Samples-repmat(Model.Mu{NdxComp},1,NumSamples);
        MiC=zeros(Dimension);
        for NdxSample=1:NumSamples
            MiC=MiC+Responsibilities(NdxComp,NdxSample)*Differences(:,NdxSample)* ...
                Differences(:,NdxSample)';
        end
        MiC=MiC/(Model.Pi(NdxComp)*NumSamples);
        if det(MiC)>0
            Model.C{NdxComp}=MiC;
            Model.InvC{NdxComp}=inv(MiC);
            Model.DetC{NdxComp}=det(MiC);        
        end
    end
    
    % See whether we have converged
    ContinueIt=((norm(AntPi-Model.Pi)/norm(Model.Pi))>0.001);
end
    