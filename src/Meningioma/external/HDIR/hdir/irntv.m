function [U, itstat] = irntv(S, KC, lambda, pars)
%  
% function [U, itstat] = irntv(S, KC, lambda, parameters)
%
% irntv --  Compute the minimum of a generalised TV functional 
%
%           T = || K*U - S ||^p + lambda*|| sqrt( (Dx(U))^2 + (Dy(U))^2 ) ||^q
%
%           for grayscale / color (vector) images using the IRN [1,2] algorithm.
%
%
%           [1] "Efficient Minimization Method for a Generalized Total 
%                Variation Functional"
%               IEEE Transactions on Image Processing, 2009, 18:2(322-332)
%
%           [2] "A Generalized Vector-Valued Total Variation Algorithm"
%               submmited to ICIP'09 (http://www.icip2009.org/)
%  
%           irntv.m is part of NUMIPAD (http://numipad.sf.net). NUMIPAD is
%           free software, you can redistribute it and/or modify it under the
%           terms of the GNU General Public License (version 2).
%
%           The NUMIPAD library is being developed under U.S. Government
%           contract W-7405-ENG-36 for Los Alamos National Laboratory.
%
%
% Usage:
%       [U, itstat] = irntv(S, KC, lambda, parameters)
%
% Input:
%       S           Input image
%       KC          Cell array {K, KT} of function handles for
%                   forward linear operator and its
%                   transpose. These operators act on images rather
%                   than vectors, but the transpose should be
%                   considered to be in the sense of the operator
%                   on a vectorised image. Set to [] for a
%                   denoising problem.
%       lambda      Regularisation term weighting factor
%       parameters  see irntvInputPars.m file (>>help irntvInputPars.m)
%
% Output:
%       U           Output image
%       itstat      Iteration status
%
% Authors
%   Paul Rodriguez    prodrig@pucp.edu.pe
%   Brendt Wohlberg   brendt@tmail.lanl.gov


if nargin < 4,
  pars = irntvInputPars('l1tv');
  if nargin < 3,
    lambda = 1;
    if nargin < 2,
      K = [];
    end
  end
end


%  ----------


% Check number of outputs (dflag --> display flag)
dflag = 1;
if nargout > 1,
  dflag = 0;
end

% Initialise iteration status vector
itstat = [];

% Set up status display
iterdispopen(dflag);


%  ----------


% Check wether the input image is grayscale or color
InputDims = size(S);

if( (length(InputDims) < 2) && (length(InputDims) > 3) )
  disp( sprintf('\nInput data has %d dimensions...\n', length(InputDims) ) );
  error('Not a valid grayscale/color image');
end


%  ----------


% Check argument KC
if ~isempty(KC),
  if ~(iscell(KC) && isa(KC{1}, 'function_handle') && isa(KC{2}, 'function_handle')),
   warning('argument KC must be empty or a cell array {K,KT} of function handles');
   U = [];
   return;
 end
end


%  ----------


% check for initial value of lambda
if( isempty(pars.lambda_ini) ) pars.lambda_ini = lambda; end


%  ----------


% check for sbstflg and p==2

if( (pars.sbstflg) && (pars.p==2) ) % Incongruent
  pars.sbstflg = 0;
end

%  ----------


s = S(:); % 1D representation of input image


% ===========================================================================
% ===========================================================================


% Start clock
tic;

if isempty(KC),  % Denoising problem

  %-----------------------------------------------------------------------
  %-----------------------------------------------------------------------

  % Choose the weighting scheme
  if( isempty(pars.adapt_epsF) || pars.adapt_epsF == 0 )    % Fidelity
      if pars.sbstflg
        fF = @(var1, var2, var3) fF_fixed_substitute(var1, var2, var3);
      else
        fF = @(var1, var2, var3) fF_fixed(var1, var2, var3);
      end
      fF_var3 = pars.epsF;
  else
      if pars.sbstflg
        fF = @(var1, var2, var3) fF_adapt_substitute(var1, var2, var3);
      else
        fF = @(var1, var2, var3) fF_adapt(var1, var2, var3);
      end
      fF_var3 = pars.epsF_cutoff;
  end

  if( isempty(pars.adapt_epsR) || pars.adapt_epsR == 0 )    % Regularization
    fR = @(var1, var2, var3, var4) fR_fixed( var1, var2, var3, var4);
    fR_var4 = pars.epsR;
  else
    fR = @(var1, var2, var3, var4) fR_adapt( var1, var2, var3, var4);
    fR_var4 = pars.epsR_cutoff;
  end

  %-----------------------------------------------------------------------

  if isempty(pars.U0),
    % Construct initial solution
    [u, flg, rlr, pit] = pcg(@(x) IDTD(x, InputDims, pars.lambda_ini), s, ...
                             pars.pcgtol_ini, pars.pcgitn, [], [], s);
    U = reshape(u, InputDims);
  else
    U = pars.U0;
    u = U(:);
  end

  % Iterate
  for k = 1:pars.loops,

    % Set weight matrices
    WF = fF(U-S, pars.p, fF_var3);
    WR = fR(U, InputDims, pars.q, fR_var4);
    

    % Compute relative residual of current solution
    if pars.sbstflg, % Use substitution (indirect) method
        wfs = s./vec(WF);
        v = u./vec(WF);
        relres = pcgrelres(@(x) IWDTWDW(x, InputDims, WF, WR, lambda), wfs, v);
    else        % Use standard (direct) method
        wfs = vec(WF).*s;
        relres = pcgrelres(@(x) IDTWD(x, InputDims, WF, WR, lambda), wfs, u);
    end
    pcgtol = relres/pars.rrs;

    % Compute and display functional values
    iterdisp(S, [], lambda, pars.p, pars.q, U, WF, WR, k, relres, [], [], ...
             pars.sbstflg, dflag);

    % Update current solution
    if pars.sbstflg, % Use substitution (indirect) method
        [v, flg, rlr, pit] = pcg(@(x) IWDTWDW(x, InputDims, WF, WR, lambda), ...
                                 wfs, pcgtol, pars.pcgitn,[],[],v);
        V = reshape(v, InputDims);
        u = vec(WF).*v;
    else        % Use standard (direct) method
        [u, flg, rlr, pit] = pcg(@(x) IDTWD(x, InputDims, WF, WR, lambda), ...
                                 wfs, pcgtol, pars.pcgitn, [], [], u);
    end

    U = reshape(u, InputDims);

    % Compute and display functional values and CG status
    is = iterdisp(S, [], lambda, pars.p, pars.q, U, WF, WR, ...
                  [], relres, flg, pit, pars.sbstflg, dflag);
    itstat = [itstat; [k is]];

  end

else % General inverse problem with linear operator K


  % KC is cell array of K and K^T function handles
  K = KC{1};
  KT = KC{2};
 

  %-----------------------------------------------------------------------
  %-----------------------------------------------------------------------

  % Choose the weighting scheme
  if( isempty(pars.adapt_epsF) || pars.adapt_epsF == 0 )    % Fidelity
        fF = @(var1, var2, var3) fF_fixed(var1, var2, var3);
        fF_var3 = pars.epsF;
  else
        fF = @(var1, var2, var3) fF_adapt(var1, var2, var3);
        fF_var3 = pars.epsF_cutoff;
  end

  if( isempty(pars.adapt_epsR) || pars.adapt_epsR == 0 )    % Regularization
    fR = @(var1, var2, var3, var4) fR_fixed( var1, var2, var3, var4);
    fR_var4 = pars.epsR;
  else
    fR = @(var1, var2, var3, var4) fR_adapt( var1, var2, var3, var4);
    fR_var4 = pars.epsR_cutoff;
  end

  %-----------------------------------------------------------------------

  
  if isempty(pars.U0),
    % Construct initial solution
    kts = vec(KT(S));
    [u, flg, rlr, pit] = pcg(@(x) KTKDTD(x, InputDims, K, KT, lambda), ...
                             kts, pars.pcgtol_ini, pars.pcgitn);
    U = reshape(u, InputDims);

  else
    U = pars.U0;
    u = U(:);
  end

  % Iterate
  for k = 1:pars.loops,


    % Set weight matrices
    WF = fF(K(U)-S, pars.p, fF_var3);
    WR = fR(U, InputDims, pars.q, fR_var4);


    % Compute relative residual of current solution
    wfs = vec(KT(WF.*S));
    relres = pcgrelres(@(x) KTWKDTWD(x, InputDims, K, KT, WF, WR, lambda), ...
                       wfs, u);
    pcgtol = relres/pars.rrs;

    % Compute and display functional values
    iterdisp(S, K, lambda, pars.p, pars.q, U, WF, WR, k, relres, ...
             [], [], [], dflag);

    % Update current solution
    [u, flg, rlr, pit] = pcg(@(x) KTWKDTWD(x, InputDims, K, KT, WF, WR, lambda), ...
                             wfs, pcgtol, pars.pcgitn,[],[],u);
    U = reshape(u, InputDims);

    % Compute and display functional values and CG status
    is = iterdisp(S, K, lambda, pars.p, pars.q, U, WF, WR, ...
                  [], relres, flg, pit, [], dflag);
    itstat = [itstat; is];

  end

end


% Final part of status display
iterdispclose(dflag);

return



% Vectorise image v
function u = vec(v)

  u = v(:);
  
return


% Consider operator Dy to be the y derivate of a vectorised
% image. Apply this operator to the unvectorised image and return
% a gradient image.
function u = Dy(v)

  u = [diff(v); zeros(1,size(v,2))];
    
return


% Consider operator Dy to be the y derivate of a vectorised
% image. Apply the transpose of this operator to the unvectorised
% image and return a gradient image.
function u = DyT(v)

  u0 = -v(1,:);
  u1 = -diff(v);
  u2 = v(end-1,:);
  u = [u0; u1(1:(end-1),:); u2];
    
return


% Consider operator Dx to be the x derivate of a vectorised
% image. Apply this operator to the unvectorised image and return
% a gradient image.
function u = Dx(v)

  u = [diff(v,1,2) zeros(size(v,1),1)];
    
return


% Consider operator Dx to be the x derivate of a vectorised
% image. Apply the transpose of this operator to the unvectorised
% image and return a gradient image.
function u = DxT(v)

  u = DyT(v')';

return


% Compute scalar function f_F (Fidelity weights)
function y = fF_fixed(x, p, epsf)

  absx = abs(x);
  mask = (absx >= epsf);
  y =  (2/p).*( ( (mask.*absx) + (1-mask)*epsf ).^(p-2) );


return


function y = fF_fixed_substitute(x, p, epsf)

  absx = abs(x);
  mask = (absx >= epsf);
  y =  sqrt(p/2).*( ( (mask.*absx) + (1-mask)*epsf ).^((2-p)/2) );


return

function y = fF_adapt(x, p, percentile)

  absx = abs(x);

  max_absx = max( absx(:) );
  min_absx = min( absx(:) );

  %find histogram between max and min.
  [h bin] = hist(absx(:), min_absx: (max_absx-min_absx)/999 :max_absx);
  hacc = cumsum(h);

  [dummy pos] = max( hacc/hacc(end) > percentile ); % 0 0 .. 0 1 1 .. 1
                                                    %         ^
                                                    %         | percentile
  epsf = bin(pos+1);

  mask = (absx >= epsf);
  y =  (2/p).*( ( (mask.*absx) + (1-mask)*epsf ).^(p-2) );

return


function y = fF_adapt_substitute(x, p, percentile)

  absx = abs(x);

  max_absx = max( absx(:) );
  min_absx = min( absx(:) );

  %find histogram between max and min.
  [h bin] = hist(absx(:), min_absx: (max_absx-min_absx)/999 :max_absx);
  hacc = cumsum(h);

  [dummy pos] = max( hacc/hacc(end) > percentile ); % 0 0 .. 0 1 1 .. 1
                                                    %         ^
                                                    %         | percentile
  epsf = bin(pos+1);

  mask = (absx >= epsf);
  y = sqrt(p/2).*( ( (mask.*absx) + (1-mask)*epsf ).^((2-p)/2) );

return


% Compute scalar function f_R (Regularization weights)
function y = fR_fixed(U, UDims, q, epsr)

  if( length(UDims) == 3 ) % only two possible values: 
                           % 3 (vector) or 2 (scalar)
    absx = Dx(U(:,:,1)).^2 + Dy(U(:,:,1)).^2;
    for k=2:UDims(3)
      absx = absx + Dx(U(:,:,k)).^2 + Dy(U(:,:,k)).^2;
    end
  else
    absx = Dx(U).^2 + Dy(U).^2 ;
  end

  mask = (absx >= epsr);

  y =  (2/q).*( ( (mask.*absx) + (1-mask)*epsr ).^((q-2)/2) );


return


% Compute scalar function f_R
function y = fR_adapt(U, UDims, q, percentile)

  if( length(UDims) == 3 ) % only two possible values: 
                           % 3 (vector) or 2 (scalar)
    absx = Dx(U(:,:,1)).^2 + Dy(U(:,:,1)).^2;
    for k=2:UDims(3)
      absx = absx + Dx(U(:,:,k)).^2 + Dy(U(:,:,k)).^2;
    end
  else
    absx = Dx(U).^2 + Dy(U).^2 ;
  end


  max_absx = max( absx(:) );
  min_absx = min( absx(:) );

  [h bin] = hist(absx(:), min_absx: (max_absx-min_absx)/999 :max_absx); 
  hacc = cumsum(h);

  [dummy pos] = max( hacc/hacc(end) > percentile ); % 0 0 .. 0 1 1 .. 1
                                                    %         ^
                                                    %         | percentile
  
  if( bin(pos) > 0 ) epsr = bin(pos);
  else epsr = bin(pos+1);
  end
  
  mask = (absx >= epsr);
  y =  (2/q).*( ( (mask.*absx) + (1-mask)*epsr ).^((q-2)/2) );


return



% Compute I + lambda*D_x^T*D_x + lambda*D_y^T*D_y for vectorised
% image (since called by pcg function)

function u = IDTD(v, vDims, lambda)
  
  V = reshape(v, vDims);

  if( length(vDims) == 3 ) % only two possible values: 
                           % 3 (vector) or 2 (scalar)
    N = vDims(1)*vDims(2);
    u = zeros( size(v) );

    for k=1:vDims(3)
      U = V(:,:,k) + lambda*DxT(Dx(V(:,:,k))) + lambda*DyT(Dy(V(:,:,k)));
      u(1+(k-1)*N:k*N) = U(:);
    end

  else

    U = V + lambda*DxT(Dx(V)) + lambda*DyT(Dy(V));
    u = U(:);

  end

return


%-----------------------------------------------------------------------------



% Compute KT*K + lambda*D_x^T*D_x + lambda*D_y^T*D_y for vectorised
% image (since called by pcg function)
function u = KTKDTD(v, vDims, K, KT, lambda)
 
  V = reshape(v, vDims);

  KTK_V = KT(K(V));

  if( length(vDims) == 3 )

    N = vDims(1)*vDims(2);
    u = zeros( size(v) );

    for k=1:vDims(3)
      U = KTK_V(:,:,k) + lambda*DxT(Dx(V(:,:,k))) + lambda*DyT(Dy(V(:,:,k)));
      u(1+(k-1)*N:k*N) = U(:);
    end

  else
    U = KT(K(V)) + lambda*DxT(Dx(V)) + lambda*DyT(Dy(V));
    u = U(:);
  end


return


% Compute I + lambda*D_x^T*W_R*D_x + lambda*D_y^T*W_R*D_y for vectorised image
% (since called by pcg function) 
function u = IDTWD(v, vDims, WF, WR, lambda)

  V = reshape(v, vDims);

  if( length(vDims) == 3 ) % only two possible values: 
                           % 3 (vector) or 2 (scalar)
    N = vDims(1)*vDims(2);
    u = zeros( size(v) );

    for k=1:vDims(3)
      U = WF(:,:,k).*V(:,:,k) + lambda*DxT(WR.*Dx(V(:,:,k))) + ...
          lambda*DyT(WR.*Dy(V(:,:,k)));
      u(1+(k-1)*N:k*N) = U(:);
    end

  else
    U = WF.*V + lambda*DxT(WR.*Dx(V)) + lambda*DyT(WR.*Dy(V));
    u = U(:);
  end  

return


% Compute I + lambda*W_F^(-1/2)*D_x^T*W_R*D_x*W_F^(-1/2) +
% lambda*W_F^(-1/2)*D_y^T*W_R*D_y*W_F^(-1/2) for vectorised image
% (since called by pcg function) 
function u = IWDTWDW(v, vDims, WFN2, WR, lambda)
  
  V = reshape(v, vDims);
%    WFN2 = WF.^(-1/2);

  if( length(vDims) == 3 )

    N = vDims(1)*vDims(2);
    u = zeros( size(v) );

    for k=1:vDims(3)

      U = V(:,:,k) + ...
          lambda*WFN2(:,:,k).*DxT(WR.*Dx(WFN2(:,:,k).*V(:,:,k))) + ...
          lambda*WFN2(:,:,k).*DyT(WR.*Dy(WFN2(:,:,k).*V(:,:,k)));
      u(1+(k-1)*N:k*N) = U(:);

    end

  else
    U = V + lambda*WFN2.*DxT(WR.*Dx(WFN2.*V)) + ...
        lambda*WFN2.*DyT(WR.*Dy(WFN2.*V));
    u = U(:);
  end

return


% Compute KT*WF*K + lambda*D_x^T*W_R*D_x + lambda*D_y^T*W_R*D_y for
% vectorised image (since called by pcg function) 
function u = KTWKDTWD(v, vDims, K, KT, WF, WR, lambda)
  
  V = reshape(v, vDims);

  KTWK_V = KT(WF.*K(V));

  if( length(vDims) == 3 )

    N = vDims(1)*vDims(2);
    u = zeros( size(v) );

    for k=1:vDims(3)
      U = KTWK_V(:,:,k) + lambda*DxT(WR.*Dx(V(:,:,k))) + ...
          lambda*DyT(WR.*Dy(V(:,:,k)));
      u(1+(k-1)*N:k*N) = U(:);
    end

  else
    U = KT(WF.*K(V)) + lambda*DxT(WR.*Dx(V)) + lambda*DyT(WR.*Dy(V));
    u = U(:);
  end

return


% Compute relative residual as used by pcg function
function y = pcgrelres(A,b,x)
  y = norm(A(x) - b)/norm(b);
return


% Compute generalised TV functional value
function [fnc, df, reg] = gentvfnctnl(S, K, lambda, p, q, U)

  sDims = size(S);
  s = S(:);
  u = U(:);

  if isempty(K),
    df = sum(abs(u-s).^p)/p;
  else
    df = sum(vec(abs(K(U)-S).^p))/p;
  end


  if(length(sDims) == 3) 
    dxu = Dx(U(:,:,1));
    dyu = Dy(U(:,:,1));
    tmp = dxu.^2 + dyu.^2;
    for k=2:sDims(3)
      dxu = Dx(U(:,:,k));
      dyu = Dy(U(:,:,k));
      tmp = tmp + dxu.^2 + dyu.^2;
    end
    reg = sum(abs(sqrt(tmp(:))).^q)/q;
  else
    dxu = Dx(U);
    dyu = Dy(U);
    reg = sum(abs(sqrt(vec(dxu.^2 + dyu.^2))).^q)/q;
  end
  fnc = df + lambda*reg;

return
  

% Compute weighted approximation to generalised TV functional value
function [fnc, df, reg] = gentvwtfnctnl(S, K, lambda, p, q, U, WF, WR)
  
  sDims = size(S);
  s = S(:);
  u = U(:);

  if isempty(K),
    df = sum(abs(vec(WF.^(1/2)).*(u-s)).^2)/2;
  else
    df = sum(abs(vec(WF.^(1/2)).*vec(K(U)-S)).^2)/2;
  end

  if(length(sDims) == 3) 
    dxu = Dx(U(:,:,1));
    dyu = Dy(U(:,:,1));
    tmp = vec((WR.^(1/2)).*dxu(:,:,1)).^2 + vec((WR.^(1/2)).*dyu(:,:,1)).^2;
    for k=2:sDims(3)
      dxu = Dx(U(:,:,k));
      dyu = Dy(U(:,:,k));
      tmp = tmp + vec((WR.^(1/2)).*dxu(:,:,1)).^2 + ...
            vec((WR.^(1/2)).*dyu(:,:,1)).^2;
    end
    reg = sum(abs( tmp ))/2;
  else
    dxu = Dx(U);
    dyu = Dy(U);
    reg = sum(abs([vec((WR.^(1/2)).*dxu); vec((WR.^(1/2)).*dyu)]).^2)/2;
  end

  fnc = df + lambda*reg;
  
return


% Compute weighted approximation to generalised TV functional value
% for denoising with substitution (indirect) method
function [fnc, df, reg] = gentvdnwtfnctnl(S, lambda, p, q, V, WFN2, WR)

 
  sDims = size(S);
  s = S(:);
  v = V(:);

  if( length(sDims) == 3 )      % color/vectorial image

    regV = ( (WR.^(1/2)).*Dx(WFN2(:,:,1).*V(:,:,1)) ).^2 + ...
    ( (WR.^(1/2)).*Dy(WFN2(:,:,1).*V(:,:,1)) ).^2;

    for k=2:sDims(3)
      regV = regV + ( (WR.^(1/2)).*Dx(WFN2(:,:,k).*V(:,:,k)) ).^2 + ...
      ( (WR.^(1/2)).*Dy(WFN2(:,:,k).*V(:,:,k)) ).^2;
    end

  else

    regV = ( (WR.^(1/2)).*Dx(WFN2.*V) ).^2 + ( (WR.^(1/2)).*Dy(WFN2.*V) ).^2;
  
  end

  reg = sum(regV(:));
  df = sum(abs(v - (s./vec(WFN2))).^2)/2;
  fnc = df + 0.5*lambda*reg;

return


% Open iteration status display
function iterdispopen(dspflag)
  if dspflag,
    disp('Itn  Fnc        DFid       Reg        WFnc       WDFid      WReg       RelRes    CGIt  Flg Time');
    disp('--------------------------------------------------------------------------------------------------');
  end
return


% Compute and display iteration status
function is = iterdisp(S, K, lambda, p, q, U, WF, WR, k, relres, pcgflg, pcgit, sbstflg, dspflag)

  t = toc;
  [tnp, dfnp, rnp] = gentvfnctnl(S, K, lambda, p, q, U);
  if isempty(K) && sbstflg,
    V = U./WF;
    [tnw, dfnw, rnw] = gentvdnwtfnctnl(S, lambda, p, q, V, WF, WR);
  else
    [tnw, dfnw, rnw] = gentvwtfnctnl(S, K, lambda, p, q, U, WF, WR);
  end
  if dspflag,
    kstr = '   ';
    if ~isempty(k),
      kstr = sprintf('%3d', k);
    end
    pistr = '     ';
    if ~isempty(pcgit),
      pistr = sprintf('%5d', pcgit);
    end
    pfstr = '  ';
    if ~isempty(pcgflg),
      pfstr = sprintf('%2d', pcgflg);
    end
    disp(sprintf('%s %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %s %s %8.1e', ...
                 kstr, tnp, dfnp, rnp, tnw, dfnw, rnw, relres, pistr, pfstr, t));
  end
  is = [tnp dfnp rnp tnw dfnw rnw relres pcgit pcgflg t];

return


% Close iteration status display
function iterdispclose(dspflag)
  if dspflag,
    disp('--------------------------------------------------------------------------------------------------');
  end
return

