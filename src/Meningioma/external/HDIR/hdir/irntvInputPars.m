function[irnPars] = irntvInputPars(dname)
%  
% function [irnPars] = irntv(methodName)
%
% Genrates de defaults parameters to be used in the IRN method, which
% computes the minimum of a generalised TV functional 
%
%           T = || K*U - S ||^p + lambda*|| sqrt( (Dx(U))^2 + (Dy(U))^2 ) ||^q
%
% for grayscale / color (vector) images using the IRN [1,2] algorithm.
%
%
% [1] "Efficient Minimization Method for a Generalized Total Variation 
%      Functional" IEEE Transactions on Image Processing, 2009, 18:2(322-332).
%
% [2] "A Generalized Vector-Valued Total Variation Algorithm"
%     submmited to ICIP'09 (http://www.icip2009.org/)
%
%    
% irntvInputPars.m is part of NUMIPAD (http://numipad.sf.net). NUMIPAD is free
% software, you can redistribute it and/or modify it under the terms of the 
% GNU General Public License (version 2).
%
% The NUMIPAD library is being developed under U.S. Government contract
% W-7405-ENG-36 for Los Alamos National Laboratory.
%
%
% Usage:
%       irnPars = irntvInputPars(methodName)
%
% Input:
%       methodName  'l1tv', 'l2tv'
%
% Output:
%       irnPars     Structure with parameters for the IRN method
%       
%       irnPars.p             Data fidelity term norm
%       irnPars.q             Data regularization term norm
%       irnPars.epsf          Data fidelity epsilon
%       irnPars.epsr          Regularization epsilon
%       irnPars.loops         Number of iterations
%       irnPars.U0            Initial solution
%       irnPars.lambda_ini    Regularisation term weighting factor for the 
%                             initial solution
%       irnPars.pcgtol_ini    PCG tolerance for the initail solution
%       irnPars.rrs           Relative residual scaling after each iteration
%       irnPars.sbstflg       Substitution flag. For p ~= 2 this options 
%                             improve performance. (see [1]).
%       irnPars.adapt_epsF    Auto-adapt the data fidelity epsilon
%       irnPars.adapt_epsR    Auto-adapt the data regularization epsilon
%       irnPars.epsR_cutoff   Cut off value when using the Auto-adapt. 
%                             See [1], section G.
%       irnPars.epsF_cutoff   Cut off value when using the Auto-adapt. 
%                             See [1], section G.
%  
% Example:
%       irnPars = irntvInputPars('l1tv')
%  
% Authors
%   Paul Rodriguez    prodrig@pucp.edu.pe
%   Brendt Wohlberg   brendt@tmail.lanl.gov

if(nargin == 0)
  dname='none';
end

irnPars = struct('p', [], 'q', [], 'epsR', [], 'epsF', [], ...
                 'loops', [], 'U0', [], 'lambda_ini', [], ...
                 'pcgtol_ini', [], 'pcgitn', [], ...
                 'rrs', [], 'sbstflg', [], ...
                 'adapt_epsR', [], 'adapt_epsF', [], ...
                 'epsR_cutoff', [], 'epsF_cutoff', []);


switch lower(dname)

  case{'l1tv'}
    irnPars.p       = 1;
    irnPars.q       = 1;
    irnPars.sbstflg = 1;

  case{'l2tv'}
    irnPars.p       = 2;
    irnPars.q       = 1;
    irnPars.sbstflg = 0;

end

irnPars.loops   = 5;
irnPars.pcgtol_ini = 1e-3;
irnPars.pcgitn  = 500;
irnPars.rrs     = 5;
irnPars.epsF    = 1e-2;
irnPars.epsR    = 1e-4;

irnPars.epsF_cutoff    = 0.05;
irnPars.epsR_cutoff    = 0.01;
